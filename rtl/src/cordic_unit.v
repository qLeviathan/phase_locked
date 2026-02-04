// ============================================================================
// CORDIC UNIT - Integer-Only Trigonometry
// ============================================================================
// Deterministic AI Apparatus - Patent Pending
//
// CORDIC (COordinate Rotation DIgital Computer) for exact integer arithmetic.
// Uses ONLY addition, subtraction, and bit shifts - no multipliers required.
//
// Operations:
// - Rotation mode: Compute sin(theta), cos(theta)
// - Vectoring mode: Compute atan2(y, x), magnitude
//
// All values use 30-bit fixed point (Q2.30 format).
// Target: Lattice iCE40 UP5K (UPduino 3.1)
// ============================================================================

`default_nettype none
`timescale 1ns / 1ps

module cordic_unit #(
    parameter WIDTH = 32,
    parameter ITERATIONS = 16,      // Number of CORDIC iterations
    parameter ANGLE_WIDTH = 32      // Angle precision (Q2.30)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,
    input  wire                     mode,           // 0=rotation, 1=vectoring

    // Inputs (Q2.30 fixed point)
    input  wire signed [WIDTH-1:0]  x_in,
    input  wire signed [WIDTH-1:0]  y_in,
    input  wire signed [ANGLE_WIDTH-1:0] angle_in,  // For rotation mode

    // Outputs (Q2.30 fixed point)
    output reg                      done,
    output reg  signed [WIDTH-1:0]  x_out,          // cos (rotation) or magnitude (vectoring)
    output reg  signed [WIDTH-1:0]  y_out,          // sin (rotation) or 0 (vectoring)
    output reg  signed [ANGLE_WIDTH-1:0] angle_out  // 0 (rotation) or atan2 (vectoring)
);

// ============================================================================
// CORDIC CONSTANTS (Q2.30 format, scaled by 2^30)
// ============================================================================

// Arctangent lookup table: atan(2^(-i)) * 2^30
// These are the rotation angles for each iteration
wire signed [ANGLE_WIDTH-1:0] ATAN_TABLE [0:ITERATIONS-1];

assign ATAN_TABLE[0]  = 32'sd843314857;   // atan(1)     = 45.0 deg
assign ATAN_TABLE[1]  = 32'sd497837829;   // atan(1/2)   = 26.565 deg
assign ATAN_TABLE[2]  = 32'sd263043837;   // atan(1/4)   = 14.036 deg
assign ATAN_TABLE[3]  = 32'sd133525159;   // atan(1/8)   = 7.125 deg
assign ATAN_TABLE[4]  = 32'sd67021687;    // atan(1/16)  = 3.576 deg
assign ATAN_TABLE[5]  = 32'sd33543516;    // atan(1/32)  = 1.790 deg
assign ATAN_TABLE[6]  = 32'sd16775851;    // atan(1/64)  = 0.895 deg
assign ATAN_TABLE[7]  = 32'sd8388437;     // atan(1/128) = 0.448 deg
assign ATAN_TABLE[8]  = 32'sd4194283;     // atan(1/256)
assign ATAN_TABLE[9]  = 32'sd2097149;     // atan(1/512)
assign ATAN_TABLE[10] = 32'sd1048576;     // atan(1/1024)
assign ATAN_TABLE[11] = 32'sd524288;
assign ATAN_TABLE[12] = 32'sd262144;
assign ATAN_TABLE[13] = 32'sd131072;
assign ATAN_TABLE[14] = 32'sd65536;
assign ATAN_TABLE[15] = 32'sd32768;

// CORDIC gain K = product of cos(atan(2^(-i))) for all i
// K ~= 0.6072529350 * 2^30 = 652032874
localparam signed [WIDTH-1:0] CORDIC_K = 32'sd652032874;

// Pi and 2*Pi in Q2.30
localparam signed [ANGLE_WIDTH-1:0] PI_SCALED     = 32'sd3373259426;
localparam signed [ANGLE_WIDTH-1:0] TWO_PI_SCALED = 32'sd6746518852;
localparam signed [ANGLE_WIDTH-1:0] HALF_PI       = 32'sd1686629713;

// ============================================================================
// STATE MACHINE
// ============================================================================
localparam S_IDLE      = 3'd0;
localparam S_PRESCALE  = 3'd1;
localparam S_ITERATE   = 3'd2;
localparam S_POSTSCALE = 3'd3;
localparam S_DONE      = 3'd4;

reg [2:0] state;

// Working registers
reg signed [WIDTH-1:0]       x_reg, y_reg;
reg signed [ANGLE_WIDTH-1:0] z_reg;
reg [4:0]                    iteration;
reg                          mode_reg;
reg signed [WIDTH-1:0]       x_prescaled, y_prescaled;

// ============================================================================
// CORDIC ITERATION LOGIC
// ============================================================================
wire signed [WIDTH-1:0] x_shifted, y_shifted;
wire signed [WIDTH-1:0] x_next, y_next;
wire signed [ANGLE_WIDTH-1:0] z_next;
wire direction;

// Shift by iteration count
assign x_shifted = x_reg >>> iteration;
assign y_shifted = y_reg >>> iteration;

// Direction of rotation
// Rotation mode: rotate towards z=0, so direction = sign(z)
// Vectoring mode: rotate y towards 0, so direction = sign(-y)
assign direction = mode_reg ? (y_reg[WIDTH-1]) : (~z_reg[ANGLE_WIDTH-1]);

// Next values based on direction
// If direction=1 (positive): rotate counter-clockwise
//   x_next = x - y_shifted
//   y_next = y + x_shifted
//   z_next = z - atan
// If direction=0 (negative): rotate clockwise
//   x_next = x + y_shifted
//   y_next = y - x_shifted
//   z_next = z + atan

assign x_next = direction ? (x_reg - y_shifted) : (x_reg + y_shifted);
assign y_next = direction ? (y_reg + x_shifted) : (y_reg - x_shifted);
assign z_next = direction ? (z_reg - ATAN_TABLE[iteration]) : (z_reg + ATAN_TABLE[iteration]);

// ============================================================================
// STATE MACHINE: Sequential Logic
// ============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        done <= 1'b0;
        x_out <= 32'd0;
        y_out <= 32'd0;
        angle_out <= 32'd0;
        x_reg <= 32'd0;
        y_reg <= 32'd0;
        z_reg <= 32'd0;
        iteration <= 5'd0;
        mode_reg <= 1'b0;
        x_prescaled <= 32'd0;
        y_prescaled <= 32'd0;
    end else begin
        case (state)
            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    mode_reg <= mode;

                    if (mode == 1'b0) begin
                        // Rotation mode: start with (K, 0) and angle
                        // Actually, we apply K at the end for better precision
                        x_reg <= 32'sd1073741824;  // 1.0 in Q2.30
                        y_reg <= 32'sd0;
                        z_reg <= angle_in;
                    end else begin
                        // Vectoring mode: start with (x, y) and compute angle
                        x_reg <= x_in;
                        y_reg <= y_in;
                        z_reg <= 32'sd0;
                    end

                    iteration <= 5'd0;
                    state <= S_ITERATE;
                end
            end

            S_ITERATE: begin
                if (iteration < ITERATIONS) begin
                    x_reg <= x_next;
                    y_reg <= y_next;
                    z_reg <= z_next;
                    iteration <= iteration + 5'd1;
                end else begin
                    state <= S_POSTSCALE;
                end
            end

            S_POSTSCALE: begin
                // Apply CORDIC gain correction
                // For rotation mode: multiply by K
                // Use shift-add approximation: K ~= 0.6072529350
                // K ~= 1/2 + 1/8 + 1/64 + 1/512 = 0.607421875

                if (mode_reg == 1'b0) begin
                    // Rotation mode: scale output by K
                    x_out <= (x_reg >>> 1) + (x_reg >>> 3) + (x_reg >>> 6) + (x_reg >>> 9);
                    y_out <= (y_reg >>> 1) + (y_reg >>> 3) + (y_reg >>> 6) + (y_reg >>> 9);
                    angle_out <= z_reg;
                end else begin
                    // Vectoring mode: magnitude in x_out, angle in angle_out
                    x_out <= (x_reg >>> 1) + (x_reg >>> 3) + (x_reg >>> 6) + (x_reg >>> 9);
                    y_out <= 32'd0;
                    angle_out <= z_reg;
                end

                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1'b1;
                state <= S_IDLE;
            end
        endcase
    end
end

endmodule

// ============================================================================
// CORDIC SIN/COS WRAPPER
// ============================================================================
module cordic_sincos #(
    parameter WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire signed [WIDTH-1:0]  angle,      // Q2.30 format

    output wire                     done,
    output wire signed [WIDTH-1:0]  sin_out,    // Q2.30 format
    output wire signed [WIDTH-1:0]  cos_out     // Q2.30 format
);

wire signed [WIDTH-1:0] x_result, y_result;
wire signed [WIDTH-1:0] angle_result;

cordic_unit #(
    .WIDTH(WIDTH),
    .ITERATIONS(16),
    .ANGLE_WIDTH(WIDTH)
) cordic_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .mode(1'b0),            // Rotation mode
    .x_in(32'd0),           // Unused in rotation
    .y_in(32'd0),           // Unused in rotation
    .angle_in(angle),
    .done(done),
    .x_out(x_result),       // cos(angle)
    .y_out(y_result),       // sin(angle)
    .angle_out(angle_result)
);

assign cos_out = x_result;
assign sin_out = y_result;

endmodule

// ============================================================================
// CORDIC ATAN2 WRAPPER
// ============================================================================
module cordic_atan2 #(
    parameter WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire signed [WIDTH-1:0]  y_in,       // Q2.30 format
    input  wire signed [WIDTH-1:0]  x_in,       // Q2.30 format

    output wire                     done,
    output wire signed [WIDTH-1:0]  angle_out,  // Q2.30 format (atan2 result)
    output wire signed [WIDTH-1:0]  magnitude   // Q2.30 format
);

cordic_unit #(
    .WIDTH(WIDTH),
    .ITERATIONS(16),
    .ANGLE_WIDTH(WIDTH)
) cordic_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .mode(1'b1),            // Vectoring mode
    .x_in(x_in),
    .y_in(y_in),
    .angle_in(32'd0),       // Unused in vectoring
    .done(done),
    .x_out(magnitude),      // sqrt(x^2 + y^2)
    .y_out(),               // ~0
    .angle_out(angle_out)   // atan2(y, x)
);

endmodule

`default_nettype wire
