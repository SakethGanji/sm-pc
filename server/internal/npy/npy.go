// Package npy reads the minimal subset of the NumPy .npy format the trainer
// emits: version 1.0, little-endian float64 ('<f8'), C order, 1-D or 2-D.
package npy

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

var shapeRe = regexp.MustCompile(`'shape':\s*\(([0-9, ]*)\)`)

// Read returns the flat data and shape (len 1 or 2).
func Read(path string) ([]float64, []int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if len(b) < 10 || string(b[:6]) != "\x93NUMPY" {
		return nil, nil, fmt.Errorf("%s: not an npy file", path)
	}
	headerLen := int(binary.LittleEndian.Uint16(b[8:10]))
	header := string(b[10 : 10+headerLen])
	if !strings.Contains(header, "'<f8'") || strings.Contains(header, "'fortran_order': True") {
		return nil, nil, fmt.Errorf("%s: unsupported npy header %q", path, header)
	}
	m := shapeRe.FindStringSubmatch(header)
	if m == nil {
		return nil, nil, fmt.Errorf("%s: no shape in header", path)
	}
	var shape []int
	for _, part := range strings.Split(m[1], ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, nil, err
		}
		shape = append(shape, n)
	}
	data := b[10+headerLen:]
	total := 1
	for _, s := range shape {
		total *= s
	}
	if len(data) < 8*total {
		return nil, nil, fmt.Errorf("%s: truncated data", path)
	}
	out := make([]float64, total)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
	}
	return out, shape, nil
}

// ReadMatrix returns data as rows x cols.
func ReadMatrix(path string) ([][]float64, error) {
	flat, shape, err := Read(path)
	if err != nil {
		return nil, err
	}
	if len(shape) == 1 {
		return [][]float64{flat}, nil
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2-D, got %v", path, shape)
	}
	rows := make([][]float64, shape[0])
	for i := range rows {
		rows[i] = flat[i*shape[1] : (i+1)*shape[1]]
	}
	return rows, nil
}

// ReadVector returns 1-D data.
func ReadVector(path string) ([]float64, error) {
	flat, shape, err := Read(path)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 {
		return nil, fmt.Errorf("%s: expected 1-D, got %v", path, shape)
	}
	return flat, nil
}
