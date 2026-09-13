const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Io = std.Io;
const Allocator = std.mem.Allocator;
const mulWide = std.math.mulWide;

const fatal = @import("fatal.zig").fatal;
const oom = @import("fatal.zig").oom;

const Matrix = @This();

data: []f32,
shape: Shape,
stride: u16,

comptime {
    assert(@sizeOf(Matrix) == 24);
}

pub const Shape = struct {
    row: u16,
    col: u16,
};

pub const SubmatrixOptions = struct {
    start: u16 = 0,
    stride: u16 = 1,
    col_count: u16,

    pub fn validate(options: SubmatrixOptions, cols: u16) void {
        assert(options.col_count > 0);
        assert(options.stride > 0);
        assert(options.start < cols);
        assert(options.start + mulWide(u16, (options.col_count - 1), options.stride) < cols);
    }
};

/// Caller owns the returned matrix and must call deinit(gpa).
///
/// This function will panic and exit on OutOfMemory
///
/// Matrix zeros memory.
pub fn init(gpa: Allocator, shape: Shape) Matrix {
    assert(shape.row > 0);
    assert(shape.col > 0);

    const data = gpa.alloc(f32, mulWide(u16, shape.row, shape.col)) catch |err| oom(err);
    @memset(data, 0);

    return .{
        .data = data,
        .shape = shape,
        .stride = shape.col,
    };
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn init_from_slice(gpa: Allocator, data: []const f32, shape: Shape) Matrix {
    assert(data.len == @as(usize, shape.row) * shape.col);

    var matrix = Matrix.init(gpa, shape);
    errdefer matrix.deinit();
    @memcpy(matrix.data, data);

    return matrix;
}

pub fn deinit(matrix: *Matrix, gpa: Allocator) void {
    matrix.assert_matrix();
    gpa.free(matrix.data);
    matrix.* = undefined;
}

inline fn assert_matrix(matrix: Matrix) void {
    assert(matrix.shape.row > 0);
    assert(matrix.shape.col > 0);
    assert(matrix.stride >= matrix.shape.col);

    const last =
        (@as(usize, matrix.shape.row) - 1) * matrix.stride +
        (@as(usize, matrix.shape.col) - 1);

    assert(matrix.data.len > last);
}

inline fn assert_shape(matrix: Matrix, row: usize, col: usize) void {
    assert(row < matrix.shape.row);
    assert(col < matrix.shape.col);
}

inline fn index(matrix: Matrix, row: usize, col: usize) usize {
    matrix.assert_matrix();
    matrix.assert_shape(row, col);
    return row * matrix.stride + col;
}

pub fn ptr(matrix: Matrix, row: usize, col: usize) *f32 {
    return &matrix.data[matrix.index(row, col)];
}

pub fn at(matrix: Matrix, row: usize, col: usize) f32 {
    return matrix.data[matrix.index(row, col)];
}

pub fn copy(target: *Matrix, source: Matrix) void {
    source.assert_matrix();
    target.assert_matrix();

    assert(target.shape.row == source.shape.row);
    assert(target.shape.col == source.shape.col);

    for (0..source.shape.row) |r| {
        for (0..source.shape.col) |c| {
            target.ptr(r, c).* = source.at(r, c);
        }
    }
}

pub fn row_view(source: Matrix, row: usize) Matrix {
    source.assert_matrix();
    assert(row < source.shape.row);

    const lo = @as(usize, row) * source.stride;
    const hi = lo + source.shape.col;

    return .{
        .data = source.data[lo..hi],
        .shape = .{
            .row = 1,
            .col = source.shape.col,
        },
        .stride = source.stride,
    };
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_row(source: Matrix, gpa: Allocator, row: usize) Matrix {
    source.assert_matrix();
    assert(row < source.shape.row);

    var target: Matrix = .init(gpa, .{
        .row = 1,
        .col = source.shape.col,
    });
    errdefer target.deinit();

    const lo = @as(usize, row) * source.stride;
    const hi = lo + source.shape.col;

    @memcpy(
        target.data[0..source.shape.col],
        source.data[lo..hi],
    );
    return target;
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_cols(source: Matrix, gpa: Allocator, options: SubmatrixOptions) Matrix {
    source.assert_matrix();
    options.validate(source.shape.col);

    var target: Matrix = .init(gpa, .{
        .row = source.shape.row,
        .col = options.col_count,
    });
    errdefer target.deinit();

    for (0..source.shape.row) |row| {
        for (0..options.col_count) |col| {
            const start: usize = @intCast(options.start);
            const stride: usize = @intCast(options.stride);

            const new_col = start + (col * stride);
            target.ptr(row, col).* = source.at(row, new_col);
        }
    }
    return target;
}

pub fn transpose(target: *Matrix, source: Matrix) void {
    source.assert_matrix();
    target.assert_matrix();
    assert(target.shape.row == source.shape.col);
    assert(target.shape.col == source.shape.row);

    for (0..target.shape.row) |r| {
        for (0..target.shape.col) |c| {
            target.ptr(r, c).* = source.at(c, r);
        }
    }
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_transpose(source: Matrix, gpa: Allocator) Matrix {
    source.assert_matrix();

    var target: Matrix = .init(gpa, .{
        .row = source.shape.col,
        .col = source.shape.row,
    });
    errdefer target.deinit();

    target.transpose(source);
    return target;
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_fill(source: Matrix, gpa: Allocator, number: f32) Matrix {
    source.assert_matrix();

    var target: Matrix = .init(gpa, .{
        .row = source.shape.row,
        .col = source.shape.col,
    });
    errdefer target.deinit();

    target.fill(number);
    return target;
}

pub fn fill(source: *Matrix, number: f32) void {
    source.assert_matrix();
    for (0..source.shape.row) |r| {
        const lo = r * source.stride;
        const hi = lo + source.shape.col;
        @memset(source.data[lo..hi], number);
    }
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_fill_random(source: Matrix, gpa: Allocator, random: std.Random) Matrix {
    source.assert_matrix();

    var target: Matrix = .init(gpa, .{
        .row = source.shape.row,
        .col = source.shape.col,
    });
    errdefer target.deinit();

    target.fill_random(random);
    return target;
}

pub fn fill_random(source: *Matrix, random: std.Random) void {
    source.assert_matrix();
    for (0..source.shape.row) |r| {
        const lo = r * source.stride;
        const hi = lo + source.shape.col;
        for (source.data[lo..hi]) |*value| {
            value.* = random.float(f32);
        }
    }
}

pub fn scale(target: *Matrix, source: Matrix, scalar: f32) void {
    source.assert_matrix();
    target.assert_matrix();
    assert(target.shape.row == source.shape.row);
    assert(target.shape.col == source.shape.col);

    for (0..source.shape.row) |r| {
        for (0..source.shape.col) |c| {
            target.ptr(r, c).* = scalar * source.at(r, c);
        }
    }
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn copy_scale(matrix: Matrix, gpa: Allocator, scalar: f32) Matrix {
    matrix.assert_matrix();

    var result: Matrix = .init(gpa, matrix.shape);
    errdefer result.deinit();

    result.scale(matrix, scalar);
    return result;
}

pub fn add_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
    res.assert_matrix();
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);
    assert(res.shape.row == mt1.shape.row);
    assert(res.shape.col == mt1.shape.col);

    for (0..mt1.shape.row) |r| {
        for (0..mt2.shape.col) |c| {
            res.ptr(r, c).* = mt1.at(r, c) + mt2.at(r, c);
        }
    }
}

pub fn sub_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
    res.assert_matrix();
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);
    assert(res.shape.row == mt1.shape.row);
    assert(res.shape.col == mt1.shape.col);

    for (0..mt1.shape.row) |r| {
        for (0..mt2.shape.col) |c| {
            res.ptr(r, c).* = mt1.at(r, c) - mt2.at(r, c);
        }
    }
}

pub fn mul_into(res: *Matrix, mt1: Matrix, mt2: Matrix) void {
    res.assert_matrix();
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.col == mt2.shape.row);
    assert(res.shape.row == mt1.shape.row);
    assert(res.shape.col == mt2.shape.col);

    for (0..mt1.shape.row) |i| {
        for (0..mt2.shape.col) |j| {
            var sum: f32 = 0;
            for (0..mt1.shape.col) |k| {
                sum += mt1.at(i, k) * mt2.at(k, j);
            }
            res.ptr(i, j).* = sum;
        }
    }
}

pub fn format(matrix: Matrix, writer: *std.Io.Writer) !void {
    matrix.assert_matrix();

    for (0..matrix.shape.row) |r| {
        const lo = r * matrix.stride;
        const hi = lo + matrix.shape.col;

        try writer.print("    .{any:.8}", .{matrix.data[lo..hi]});
        if (r + 1 < matrix.shape.row) try writer.print(",", .{});
        try writer.print("\n", .{});
    }
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn add(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);

    var res: Matrix = .init(gpa, mt1.shape);
    errdefer res.deinit();
    res.add_into(mt1, mt2);
    return res;
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn sub(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.row == mt2.shape.row);
    assert(mt1.shape.col == mt2.shape.col);

    var res: Matrix = .init(gpa, mt1.shape);
    errdefer res.deinit();
    res.sub_into(mt1, mt2);
    return res;
}

/// Caller owns the returned matrix and must call deinit(gpa).
pub fn mul(gpa: Allocator, mt1: Matrix, mt2: Matrix) Matrix {
    mt1.assert_matrix();
    mt2.assert_matrix();
    assert(mt1.shape.col == mt2.shape.row);

    var res: Matrix = .init(gpa, .{
        .row = mt1.shape.row,
        .col = mt2.shape.col,
    });
    errdefer res.deinit();
    res.mul_into(mt1, mt2);
    return res;
}

fn expect_approx_slices(comptime T: type, expected: []const T, actual: []const T) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |exp, act, i| {
        testing.expectApproxEqRel(exp, act, 0.0001) catch |err| {
            std.debug.print(
                \\
                \\Mismatch at slice index {d}:
                \\    expected: {d}
                \\      actual: {d}
                \\
            , .{ i, exp, act });
            return err;
        };
    }
}

test "init matrix" {
    var matrix: Matrix = .init(testing.allocator, .{
        .row = 3,
        .col = 4,
    });
    defer matrix.deinit(testing.allocator);
    matrix.fill(2);

    try testing.expectEqual(3, matrix.shape.row);
    try testing.expectEqual(4, matrix.shape.col);
    try testing.expectEqual(4, matrix.stride);
    try testing.expectEqual(12, matrix.data.len);
    std.log.debug("matrix:\n{f}", .{matrix});
}

test "fill_random produces varying values" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    var matrix: Matrix = .init(gpa, .{
        .row = 4,
        .col = 4,
    });
    defer matrix.deinit(gpa);

    matrix.fill_random(random);
    var all_equal = true;
    for (matrix.data[1..]) |value| {
        if (value != matrix.data[0]) {
            all_equal = false;
            break;
        }
    }
    try testing.expect(!all_equal);
}

test "smoke test" {
    var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();

    var mt1: Matrix = .init(arena, .{ .row = 3, .col = 3 });
    var mt2: Matrix = .init(arena, .{ .row = 3, .col = 3 });
    mt1.fill(4);
    mt2.fill(2);
    const mt1_scaled = mt1.copy_scale(arena, 2);

    for (0..3) |r| {
        for (0..3) |c| {
            try testing.expect(mt1_scaled.at(r, c) == mt1_scaled.ptr(r, c).*);
        }
    }

    const sum = Matrix.add(arena, mt1, mt2);
    const diff = Matrix.sub(arena, mt1, mt2);
    const prod = Matrix.mul(arena, mt1, mt2);
    const mt1_row_vector = mt1.copy_row(arena, 2);
    const mt1_submatrix = mt1.copy_cols(arena, .{ .col_count = 2 });
    const mt1_transpose = mt1.copy_transpose(arena);

    var mtx_test: Matrix = .init(arena, .{ .row = 3, .col = 3 });
    try expect_approx_slices(
        f32,
        mt1_scaled.data,
        mtx_test.copy_fill(arena, 8).data,
    );
    try expect_approx_slices(
        f32,
        sum.data,
        mtx_test.copy_fill(arena, 6).data,
    );
    try expect_approx_slices(
        f32,
        diff.data,
        mtx_test.copy_fill(arena, 2).data,
    );
    try expect_approx_slices(
        f32,
        prod.data,
        mtx_test.copy_fill(arena, 24).data,
    );
    try expect_approx_slices(
        f32,
        mt1_row_vector.data,
        Matrix.init(arena, .{ .row = 1, .col = 3 }).copy_fill(arena, 4).data,
    );
    try expect_approx_slices(
        f32,
        mt1_submatrix.data,
        Matrix.init(arena, .{ .row = 3, .col = 2 }).copy_fill(arena, 4).data,
    );
    try expect_approx_slices(
        f32,
        mt1_transpose.data,
        mtx_test.copy_fill(arena, 4).data,
    );

    var data = [_]f32{
        0, 0, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    };
    const matrix: Matrix = .init_from_slice(arena, &data, .{ .row = 4, .col = 3 });

    const matrix_row = matrix.copy_row(arena, 2);
    try expect_approx_slices(
        f32,
        matrix_row.data,
        &.{ 1, 0, 0 },
    );

    const matrix_cols = matrix.copy_cols(arena, .{ .col_count = 2, .stride = 2 });
    try expect_approx_slices(
        f32,
        matrix_cols.data,
        &.{ 0, 0, 0, 0, 1, 0, 1, 1 },
    );
}

test "property based fuzzing" {
    const T = struct {
        fn fuzz_test(_: void, smith: *testing.Smith) !void {
            var arena_instance: std.heap.ArenaAllocator = .init(testing.allocator);
            defer arena_instance.deinit();
            const arena = arena_instance.allocator();
            var prng: std.Random.DefaultPrng = .init(testing.random_seed);
            const random = prng.random();

            while (!smith.eos()) {
                defer _ = arena_instance.reset(.free_all);
                const row = smith.value(u16) % 8 + 1;
                const col = smith.value(u16) % 8 + 1;

                var A: Matrix = .init(arena, .{ .row = row, .col = col });
                var B: Matrix = .init(arena, .{ .row = row, .col = col });
                var C: Matrix = .init(arena, .{ .row = row, .col = col });
                A.fill_random(random);
                B.fill_random(random);
                C.fill_random(random);

                // For mul
                const m = smith.value(u16) % 8 + 1;
                const n = smith.value(u16) % 8 + 1;
                const p = smith.value(u16) % 8 + 1;
                const q = smith.value(u16) % 8 + 1;

                var D: Matrix = .init(arena, .{ .row = m, .col = n });
                var E: Matrix = .init(arena, .{ .row = n, .col = p });
                var F: Matrix = .init(arena, .{ .row = p, .col = q });
                D.fill_random(random);
                E.fill_random(random);
                F.fill_random(random);

                // A + B = B + A
                try expect_approx_slices(
                    f32,
                    Matrix.add(arena, A, B).data,
                    Matrix.add(arena, B, A).data,
                );

                // A + (B + C) = (A + B) + C
                try expect_approx_slices(
                    f32,
                    Matrix.add(arena, A, .add(arena, B, C)).data,
                    Matrix.add(arena, .add(arena, A, B), C).data,
                );

                // (DE)F = D(EF)
                try expect_approx_slices(
                    f32,
                    Matrix.mul(arena, .mul(arena, D, E), F).data,
                    Matrix.mul(arena, D, .mul(arena, E, F)).data,
                );

                // (A^T)^T = A
                try expect_approx_slices(
                    f32,
                    A.copy_transpose(arena).copy_transpose(arena).data,
                    A.data,
                );

                // (A + B)^T = A^T + B^T
                try expect_approx_slices(
                    f32,
                    Matrix.add(arena, A, B).copy_transpose(arena).data,
                    Matrix.add(arena, A.copy_transpose(arena), B.copy_transpose(arena)).data,
                );

                // (DE)^T = (E^T)(D^T)
                try expect_approx_slices(
                    f32,
                    Matrix.mul(arena, D, E).copy_transpose(arena).data,
                    Matrix.mul(arena, E.copy_transpose(arena), D.copy_transpose(arena)).data,
                );

                // r(A)^T = (rA)^T
                const r = smith.value(u8);
                try expect_approx_slices(
                    f32,
                    A.copy_transpose(arena).copy_scale(arena, r).data,
                    A.copy_scale(arena, r).copy_transpose(arena).data,
                );

                var X: Matrix = .init(arena, .{ .row = row, .col = row });
                var Y: Matrix = .init(arena, .{ .row = row, .col = row });
                var Z: Matrix = .init(arena, .{ .row = row, .col = row });
                X.fill_random(random);
                Y.fill_random(random);
                Z.fill_random(random);

                // X(Y + Z) = XY + XZ
                try expect_approx_slices(
                    f32,
                    Matrix.mul(arena, X, .add(arena, Y, Z)).data,
                    Matrix.add(arena, .mul(arena, X, Y), .mul(arena, X, Z)).data,
                );
            }
        }
    };

    try testing.fuzz({}, T.fuzz_test, .{});
}
