// ============================================================================
// CASCADE OPERATOR (KAPPA)
// ============================================================================
// Deterministic AI Apparatus - Patent Pending
//
// Implements the cascade operator kappa that resolves adjacent 1s in
// Zeckendorf representation: 11 -> 100 (based on F[k] + F[k+1] = F[k+2])
//
// This is the core rewrite operation that maintains canonical form.
// Purely combinational for single-cycle execution.
//
// Target: Lattice iCE40 UP5K (UPduino 3.1)
// ============================================================================

`default_nettype none
`timescale 1ns / 1ps

module cascade_operator #(
    parameter WIDTH = 32
)(
    input  wire [WIDTH-1:0]     bits_in,
    output wire [WIDTH-1:0]     bits_out,
    output wire                 valid,          // No adjacent 1s
    output wire                 changed,        // Cascade was applied
    output wire [4:0]           cascade_pos     // Position of cascade (if any)
);

// ============================================================================
// DETECT ADJACENT 1s
// ============================================================================
wire [WIDTH-1:0] adjacent;
assign adjacent = bits_in & (bits_in << 1);

// Find lowest adjacent pair position
// adjacent[k] = 1 means bits_in[k] and bits_in[k-1] are both 1
wire [WIDTH-1:0] lowest_adjacent;
assign lowest_adjacent = adjacent & (~adjacent + 1);  // Isolate lowest bit

// Decode position
wire [4:0] pos;
assign pos = lowest_adjacent[1]  ? 5'd0  :
             lowest_adjacent[2]  ? 5'd1  :
             lowest_adjacent[3]  ? 5'd2  :
             lowest_adjacent[4]  ? 5'd3  :
             lowest_adjacent[5]  ? 5'd4  :
             lowest_adjacent[6]  ? 5'd5  :
             lowest_adjacent[7]  ? 5'd6  :
             lowest_adjacent[8]  ? 5'd7  :
             lowest_adjacent[9]  ? 5'd8  :
             lowest_adjacent[10] ? 5'd9  :
             lowest_adjacent[11] ? 5'd10 :
             lowest_adjacent[12] ? 5'd11 :
             lowest_adjacent[13] ? 5'd12 :
             lowest_adjacent[14] ? 5'd13 :
             lowest_adjacent[15] ? 5'd14 :
             lowest_adjacent[16] ? 5'd15 :
             lowest_adjacent[17] ? 5'd16 :
             lowest_adjacent[18] ? 5'd17 :
             lowest_adjacent[19] ? 5'd18 :
             lowest_adjacent[20] ? 5'd19 :
             lowest_adjacent[21] ? 5'd20 :
             lowest_adjacent[22] ? 5'd21 :
             lowest_adjacent[23] ? 5'd22 :
             lowest_adjacent[24] ? 5'd23 :
             lowest_adjacent[25] ? 5'd24 :
             lowest_adjacent[26] ? 5'd25 :
             lowest_adjacent[27] ? 5'd26 :
             lowest_adjacent[28] ? 5'd27 :
             lowest_adjacent[29] ? 5'd28 :
             lowest_adjacent[30] ? 5'd29 :
             lowest_adjacent[31] ? 5'd30 : 5'd31;

// ============================================================================
// APPLY CASCADE: 11 at (pos, pos+1) -> 100 at pos+2
// ============================================================================
wire [WIDTH-1:0] clear_mask;
wire [WIDTH-1:0] toggle_mask;

// Clear bits at pos and pos+1
assign clear_mask = (adjacent != 0) ? ~((1 << pos) | (1 << (pos + 1))) : {WIDTH{1'b1}};

// Toggle bit at pos+2
assign toggle_mask = (adjacent != 0) ? (1 << (pos + 2)) : {WIDTH{1'b0}};

// Apply cascade
assign bits_out = (bits_in & clear_mask) ^ toggle_mask;

// ============================================================================
// OUTPUTS
// ============================================================================
assign valid = (adjacent == 0);
assign changed = (adjacent != 0);
assign cascade_pos = pos;

endmodule

// ============================================================================
// ITERATIVE CASCADE (Multi-cycle for complete resolution)
// ============================================================================
module cascade_iterative #(
    parameter WIDTH = 32,
    parameter MAX_ITERATIONS = 15
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    input  wire [WIDTH-1:0]     bits_in,

    output reg                  done,
    output reg                  valid,
    output reg  [WIDTH-1:0]     bits_out,
    output reg  [3:0]           iterations
);

// Single-cycle cascade instance
wire [WIDTH-1:0] cascade_result;
wire             cascade_valid;
wire             cascade_changed;

cascade_operator #(.WIDTH(WIDTH)) cascade_inst (
    .bits_in(bits_out),
    .bits_out(cascade_result),
    .valid(cascade_valid),
    .changed(cascade_changed),
    .cascade_pos()
);

// State machine
localparam S_IDLE = 2'd0;
localparam S_CASCADE = 2'd1;
localparam S_DONE = 2'd2;

reg [1:0] state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        done <= 1'b0;
        valid <= 1'b0;
        bits_out <= {WIDTH{1'b0}};
        iterations <= 4'd0;
    end else begin
        case (state)
            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    bits_out <= bits_in;
                    iterations <= 4'd0;
                    state <= S_CASCADE;
                end
            end

            S_CASCADE: begin
                if (cascade_valid || iterations >= MAX_ITERATIONS) begin
                    state <= S_DONE;
                end else begin
                    bits_out <= cascade_result;
                    iterations <= iterations + 4'd1;
                end
            end

            S_DONE: begin
                done <= 1'b1;
                valid <= cascade_valid;
                state <= S_IDLE;
            end
        endcase
    end
end

endmodule

`default_nettype wire
