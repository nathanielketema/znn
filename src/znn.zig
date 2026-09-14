const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const Matrix = @import("Matrix.zig");
const fatal = @import("fatal.zig").fatal;
const oom = @import("fatal.zig").oom;

pub const NN = struct {
    weights: []Matrix,
    biases: []Matrix,
    deltas: []Matrix,
    activations: []Matrix,
    activation_fn: Activation,
    arena: std.heap.ArenaAllocator,

    pub const max_layers: usize = 16;

    pub const Activation = enum {
        relu,
        sigmoid,

        pub fn apply(activation_fn: Activation, matrix: *Matrix) void {
            return switch (activation_fn) {
                .relu => {
                    for (0..matrix.shape.row) |r| {
                        for (0..matrix.shape.col) |c| {
                            matrix.ptr(r, c).* = relu_fn(matrix.at(r, c));
                        }
                    }
                },
                .sigmoid => {
                    for (0..matrix.shape.row) |r| {
                        for (0..matrix.shape.col) |c| {
                            matrix.ptr(r, c).* = sigmoid_fn(matrix.at(r, c));
                        }
                    }
                },
            };
        }

        pub fn derivative(activation_fn: Activation, x: f32) f32 {
            return switch (activation_fn) {
                .relu => if (x > 0) @as(f32, 1.0) else @as(f32, 0.0),
                .sigmoid => x * (1 - x),
            };
        }

        fn relu_fn(x: f32) f32 {
            return @max(0, x);
        }

        fn sigmoid_fn(x: f32) f32 {
            return 1 / (1 + @exp(-x));
        }
    };

    /// Caller must free memory by calling deinit()
    /// This function will panic and exit on OutOfMemory
    pub fn init(gpa: Allocator, options: struct {
        architecture: []const u16,
        activation_fn: Activation = .sigmoid,
        random: std.Random,
    }) NN {
        assert(options.architecture.len >= 2);
        assert(options.architecture.len <= max_layers);
        for (options.architecture) |neurons| assert(neurons > 0);

        var arena_instance: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena_instance.deinit();
        const arena = arena_instance.allocator();

        const layer_count = options.architecture.len;
        const edge_count = layer_count - 1;

        const weights = arena.alloc(Matrix, edge_count) catch |err| oom(err);
        const biases = arena.alloc(Matrix, edge_count) catch |err| oom(err);
        const deltas = arena.alloc(Matrix, edge_count) catch |err| oom(err);
        const activations = arena.alloc(Matrix, layer_count) catch |err| oom(err);

        for (0..edge_count) |i| {
            activations[i] = .init(arena, .{
                .row = 1,
                .col = options.architecture[i],
            });
            weights[i] = .init(arena, .{
                .row = options.architecture[i],
                .col = options.architecture[i + 1],
            });
            biases[i] = .init(arena, .{
                .row = 1,
                .col = options.architecture[i + 1],
            });
            deltas[i] = .init(arena, .{ .row = 1, .col = options.architecture[i + 1] });
        }
        activations[edge_count] = .init(arena, .{
            .row = 1,
            .col = options.architecture[edge_count],
        });

        var nn: NN = .{
            .weights = weights,
            .biases = biases,
            .activations = activations,
            .deltas = deltas,
            .activation_fn = options.activation_fn,
            .arena = arena_instance,
        };
        nn.fill_rand(options.random);
        return nn;
    }

    pub fn deinit(nn: *NN) void {
        nn.arena.deinit();
    }

    pub fn fill_rand(nn: *NN, random: std.Random) void {
        for (nn.weights) |*wght| wght.fill_random(random);
        for (nn.biases) |*bias| bias.fill(0);
    }

    pub fn format(nn: NN, writer: *Io.Writer) !void {
        try writer.print("  weights = {{\n", .{});
        for (nn.weights, 0..) |wght, i| {
            if (i > 0) try writer.print("\n", .{});
            try writer.print("{f}", .{wght});
        }
        try writer.print("  }},\n", .{});

        try writer.print("  biases = {{\n", .{});
        for (nn.biases, 0..) |bias, i| {
            if (i > 0) try writer.print("\n", .{});
            try writer.print("{f}", .{bias});
        }
        try writer.print("  }},\n", .{});

        try writer.print("  activations = {{\n", .{});
        for (nn.activations, 0..) |actv, i| {
            if (i > 0) try writer.print("\n", .{});
            try writer.print("{f}", .{actv});
        }
        try writer.print("  }},\n", .{});
    }

    pub fn train(
        nn: *NN,
        inputs: Matrix,
        targets: Matrix,
        options: struct { alpha: f32, epoch: usize },
    ) void {
        assert(inputs.shape.row == targets.shape.row);
        assert(inputs.shape.col == nn.activations[0].shape.col);
        assert(targets.shape.col == nn.activations[nn.activations.len - 1].shape.col);

        for (0..options.epoch) |_| {
            for (0..inputs.shape.row) |r| {
                nn.forward(inputs.row_view(r));
                nn.backward(targets.row_view(r));
                nn.learn(options.alpha);
            }
        }
    }

    /// No allocation. Usefuly when run in a loop.
    pub fn predict_into(nn: *NN, output: *Matrix, input: Matrix) void {
        nn.forward(input);
        output.copy(nn.activations[nn.activations.len - 1]);
    }

    pub fn predict(nn: *NN, gpa: Allocator, input: Matrix) Matrix {
        const final = nn.activations[nn.activations.len - 1];
        var predicted: Matrix = .init(gpa, .{ .row = final.shape.row, .col = final.shape.col });
        nn.predict_into(&predicted, input);
        return predicted;
    }

    fn forward(nn: *NN, training_input: Matrix) void {
        nn.activations[0].copy(training_input);
        for (0..nn.weights.len) |i| {
            nn.activations[i + 1].mul_into(
                nn.activations[i],
                nn.weights[i],
            );
            nn.activations[i + 1].add_into(nn.activations[i + 1], nn.biases[i]);
            nn.activation_fn.apply(&nn.activations[i + 1]);
        }
    }

    fn backward(nn: *NN, training_target: Matrix) void {
        const output_index = nn.activations.len - 1;
        var output_layer = nn.activations[output_index];

        assert(training_target.shape.row == output_layer.shape.row);
        assert(training_target.shape.col == output_layer.shape.col);

        var output_delta = nn.deltas[output_index - 1];
        for (0..output_delta.shape.row) |r| {
            for (0..output_delta.shape.col) |c| {
                const pred = output_layer.at(r, c);
                const goal = training_target.at(r, c);

                const loss = pred - goal;
                output_delta.ptr(r, c).* = loss * nn.activation_fn.derivative(pred);
            }
        }

        var i: usize = nn.weights.len;
        while (i > 0) {
            i -= 1;

            if (i > 0) {
                for (0..nn.deltas[i - 1].shape.row) |r| {
                    for (0..nn.deltas[i - 1].shape.col) |c| {
                        const sum = b: {
                            var sum_tmp: f32 = 0;
                            for (0..nn.deltas[i].shape.col) |k| {
                                sum_tmp += nn.deltas[i].at(r, k) * nn.weights[i].at(c, k);
                            }
                            break :b sum_tmp;
                        };

                        const output = nn.activations[i].at(r, c);
                        nn.deltas[i - 1].ptr(r, c).* = sum * nn.activation_fn.derivative(output);
                    }
                }
            }
        }
    }

    fn learn(nn: *NN, alpha: f32) void {
        for (0..nn.weights.len) |i| {
            for (0..nn.weights[i].shape.row) |r| {
                for (0..nn.weights[i].shape.col) |c| {
                    const gradient = nn.activations[i].at(0, r) * nn.deltas[i].at(0, c);
                    nn.weights[i].ptr(r, c).* -= alpha * gradient;
                }
            }

            for (0..nn.biases[i].shape.row) |r| {
                for (0..nn.biases[i].shape.col) |c| {
                    nn.biases[i].ptr(r, c).* -= alpha * nn.deltas[i].at(r, c);
                }
            }
        }
    }
};

test "smoke" {
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    var nn: NN = .init(testing.allocator, .{
        .architecture = &.{ 2, 3, 1 },
        .random = random,
    });
    defer nn.deinit();
}

test "xor" {
    const gpa = testing.allocator;
    var arena_instance: std.heap.ArenaAllocator = .init(gpa);
    defer arena_instance.deinit();
    const arena = arena_instance.allocator();
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    var xor = [_]f32{
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    };
    var xor_matrix: Matrix = .init_from_slice(arena, &xor, .{ .row = 4, .col = 3 });
    var inputs = xor_matrix.copy_cols(arena, .{ .col_count = 2 });
    const targets = xor_matrix.copy_cols(arena, .{ .col_count = 1, .start = 2 });
    var nn: NN = .init(gpa, .{
        .architecture = &.{ 2, 2, 1 },
        .activation_fn = .sigmoid,
        .random = random,
    });
    defer nn.deinit();

    nn.train(inputs, targets, .{ .alpha = 1, .epoch = 5000 });

    var output: Matrix = .init(arena, .{ .row = 1, .col = 1 });
    for (0..inputs.shape.row) |r| {
        nn.predict_into(&output, inputs.row_view(r));
        // _ = nn.predict(arena, inputs.row_view(r));
        // std.debug.print("{d} ^ {d} = {d}, {f}", .{
        //     inputs.at(r, 0),
        //     inputs.at(r, 1),
        //     targets.at(r, 0),
        //     nn.predict(arena, inputs.row_view(r)),
        // });
    }
}
