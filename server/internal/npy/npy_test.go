package npy

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// write a v1.0 npy file the way numpy does for float64 C-order.
func writeNpy(t *testing.T, path, shape string, vals []float64) {
	t.Helper()
	header := "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape + ", }"
	for (10+len(header)+1)%64 != 0 {
		header += " "
	}
	header += "\n"
	buf := append([]byte("\x93NUMPY\x01\x00"), byte(len(header)), byte(len(header)>>8))
	buf = append(buf, header...)
	for _, v := range vals {
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(v))
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestReadMatrixAndVector(t *testing.T) {
	dir := t.TempDir()
	mpath := filepath.Join(dir, "m.npy")
	writeNpy(t, mpath, "(2, 3)", []float64{1, 2, 3, 4, 5, 6})
	m, err := ReadMatrix(mpath)
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 || m[1][2] != 6 || m[0][1] != 2 {
		t.Fatalf("%v", m)
	}

	vpath := filepath.Join(dir, "v.npy")
	writeNpy(t, vpath, "(3,)", []float64{0.5, -1.25, 7})
	v, err := ReadVector(vpath)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || v[1] != -1.25 {
		t.Fatalf("%v", v)
	}
}

func TestRejectsUnsupportedVersionAndTruncation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.npy")
	writeNpy(t, path, "(3,)", []float64{1, 2, 3})
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	v2 := append([]byte(nil), b...)
	v2[6] = 2 // v2.0 uses a 4-byte header length — must not be parsed as v1
	if err := os.WriteFile(path, v2, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Read(path); err == nil {
		t.Fatal("expected error for npy version 2.0")
	}

	if err := os.WriteFile(path, b[:12], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Read(path); err == nil {
		t.Fatal("expected error for truncated header")
	}
}
