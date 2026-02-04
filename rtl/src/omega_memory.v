// ============================================================================
// OMEGA-INDEXED CONTENT-ADDRESSABLE MEMORY
// ============================================================================
// Deterministic AI Apparatus - Patent Pending
//
// Implements holographic memory using Omega-indexing for content-addressable
// storage and retrieval. Omega = sum of active Fibonacci indices.
//
// Memory organization:
// - Primary index: Omega value (0-63)
// - Secondary: Hash of Zeckendorf bit pattern
// - Uses iCE40 UP5K SPRAM (256Kbit blocks)
//
// Target: Lattice iCE40 UP5K (UPduino 3.1)
// ============================================================================

`default_nettype none
`timescale 1ns / 1ps

// ============================================================================
// iCE40 SPRAM Primitive Wrapper
// ============================================================================
module ice40_spram_256k (
    input  wire        clk,
    input  wire        we,
    input  wire [13:0] addr,
    input  wire [15:0] data_in,
    output wire [15:0] data_out
);

`ifdef SYNTHESIS
    // Instantiate actual SPRAM primitive for synthesis
    SB_SPRAM256KA spram_inst (
        .ADDRESS(addr),
        .DATAIN(data_in),
        .MASKWREN(4'b1111),
        .WREN(we),
        .CHIPSELECT(1'b1),
        .CLOCK(clk),
        .STANDBY(1'b0),
        .SLEEP(1'b0),
        .POWEROFF(1'b1),
        .DATAOUT(data_out)
    );
`else
    // Behavioral model for simulation
    reg [15:0] mem [0:16383];
    reg [15:0] data_out_reg;

    always @(posedge clk) begin
        if (we) begin
            mem[addr] <= data_in;
        end
        data_out_reg <= mem[addr];
    end

    assign data_out = data_out_reg;
`endif

endmodule

// ============================================================================
// OMEGA MEMORY CONTROLLER
// ============================================================================
module omega_memory #(
    parameter ZECK_WIDTH = 32,
    parameter ENERGY_WIDTH = 32,
    parameter THETA_WIDTH = 32,
    parameter OMEGA_WIDTH = 6,      // Omega values 0-63
    parameter ENTRIES_PER_OMEGA = 16 // Max entries per Omega bucket
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Store interface
    input  wire                     store_valid,
    input  wire [ZECK_WIDTH-1:0]    store_zeck,
    input  wire [ENERGY_WIDTH-1:0]  store_energy,
    input  wire [THETA_WIDTH-1:0]   store_theta,
    input  wire [OMEGA_WIDTH-1:0]   store_omega,
    output reg                      store_done,

    // Recall interface (exact match)
    input  wire                     recall_valid,
    input  wire [OMEGA_WIDTH-1:0]   recall_omega,
    output reg                      recall_done,
    output reg                      recall_found,
    output reg  [ZECK_WIDTH-1:0]    recall_zeck,
    output reg  [ENERGY_WIDTH-1:0]  recall_energy,
    output reg  [THETA_WIDTH-1:0]   recall_theta,

    // Similarity search interface
    input  wire                     search_valid,
    input  wire [OMEGA_WIDTH-1:0]   search_omega,
    input  wire [2:0]               search_tolerance,   // +/- tolerance
    output reg                      search_done,
    output reg  [3:0]               search_count,       // Number of matches
    output reg  [ZECK_WIDTH-1:0]    search_best_zeck,
    output reg  [ENERGY_WIDTH-1:0]  search_best_energy
);

// ============================================================================
// MEMORY ORGANIZATION
// ============================================================================
// Address = {omega[5:0], entry_index[3:0]} = 10 bits
// Each entry = {zeck[31:0], energy[31:0], theta[31:0]} = 96 bits = 6x16-bit words
//
// SPRAM provides 256Kbit = 16K x 16-bit words
// We use 64 omega buckets x 16 entries x 6 words = 6144 words (fits easily)

localparam ENTRY_WORDS = 6;  // 96 bits per entry
localparam BUCKET_SIZE = ENTRIES_PER_OMEGA * ENTRY_WORDS;

// State machine
localparam S_IDLE        = 4'd0;
localparam S_STORE_READ  = 4'd1;
localparam S_STORE_WRITE = 4'd2;
localparam S_STORE_NEXT  = 4'd3;
localparam S_RECALL_READ = 4'd4;
localparam S_RECALL_CHECK= 4'd5;
localparam S_SEARCH_INIT = 4'd6;
localparam S_SEARCH_READ = 4'd7;
localparam S_SEARCH_EVAL = 4'd8;
localparam S_DONE        = 4'd9;

reg [3:0] state;

// Working registers
reg [OMEGA_WIDTH-1:0]   work_omega;
reg [3:0]               work_entry;
reg [2:0]               work_word;
reg [ZECK_WIDTH-1:0]    work_zeck;
reg [ENERGY_WIDTH-1:0]  work_energy;
reg [THETA_WIDTH-1:0]   work_theta;
reg [2:0]               work_tolerance;
reg signed [OMEGA_WIDTH:0] search_omega_min;
reg signed [OMEGA_WIDTH:0] search_omega_max;
reg [OMEGA_WIDTH-1:0]   search_current_omega;
reg [ENERGY_WIDTH-1:0]  best_energy;

// Entry count per omega bucket (stored in first word of each bucket)
reg [3:0] bucket_count [0:63];

// SPRAM interface
reg         spram_we;
reg  [13:0] spram_addr;
reg  [15:0] spram_data_in;
wire [15:0] spram_data_out;

ice40_spram_256k spram_inst (
    .clk(clk),
    .we(spram_we),
    .addr(spram_addr),
    .data_in(spram_data_in),
    .data_out(spram_data_out)
);

// ============================================================================
// ADDRESS CALCULATION
// ============================================================================
function [13:0] calc_addr;
    input [OMEGA_WIDTH-1:0] omega;
    input [3:0] entry;
    input [2:0] word;
    begin
        calc_addr = {omega, entry, word} + 14'd64;  // Skip header area
    end
endfunction

// ============================================================================
// INITIALIZATION
// ============================================================================
integer i;
initial begin
    for (i = 0; i < 64; i = i + 1) begin
        bucket_count[i] = 4'd0;
    end
end

// ============================================================================
// STATE MACHINE
// ============================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= S_IDLE;
        store_done <= 1'b0;
        recall_done <= 1'b0;
        recall_found <= 1'b0;
        search_done <= 1'b0;
        search_count <= 4'd0;
        spram_we <= 1'b0;
        spram_addr <= 14'd0;
        spram_data_in <= 16'd0;
        work_omega <= 6'd0;
        work_entry <= 4'd0;
        work_word <= 3'd0;
        work_zeck <= 32'd0;
        work_energy <= 32'd0;
        work_theta <= 32'd0;
        best_energy <= 32'd0;
    end else begin
        // Default
        spram_we <= 1'b0;
        store_done <= 1'b0;
        recall_done <= 1'b0;
        search_done <= 1'b0;

        case (state)
            S_IDLE: begin
                if (store_valid) begin
                    work_omega <= store_omega;
                    work_zeck <= store_zeck;
                    work_energy <= store_energy;
                    work_theta <= store_theta;
                    work_entry <= bucket_count[store_omega];
                    work_word <= 3'd0;
                    state <= S_STORE_WRITE;
                end else if (recall_valid) begin
                    work_omega <= recall_omega;
                    work_entry <= 4'd0;
                    work_word <= 3'd0;
                    recall_found <= 1'b0;
                    state <= S_RECALL_READ;
                end else if (search_valid) begin
                    work_tolerance <= search_tolerance;
                    search_omega_min <= (search_omega > search_tolerance) ?
                                        (search_omega - search_tolerance) : 0;
                    search_omega_max <= (search_omega + search_tolerance < 64) ?
                                        (search_omega + search_tolerance) : 63;
                    search_current_omega <= (search_omega > search_tolerance) ?
                                           (search_omega - search_tolerance) : 0;
                    search_count <= 4'd0;
                    best_energy <= 32'd0;
                    work_entry <= 4'd0;
                    state <= S_SEARCH_INIT;
                end
            end

            // ----------------------------------------------------------------
            // STORE: Write entry to memory
            // ----------------------------------------------------------------
            S_STORE_WRITE: begin
                spram_we <= 1'b1;
                spram_addr <= calc_addr(work_omega, work_entry, work_word);

                case (work_word)
                    3'd0: spram_data_in <= work_zeck[15:0];
                    3'd1: spram_data_in <= work_zeck[31:16];
                    3'd2: spram_data_in <= work_energy[15:0];
                    3'd3: spram_data_in <= work_energy[31:16];
                    3'd4: spram_data_in <= work_theta[15:0];
                    3'd5: spram_data_in <= work_theta[31:16];
                endcase

                state <= S_STORE_NEXT;
            end

            S_STORE_NEXT: begin
                if (work_word < 3'd5) begin
                    work_word <= work_word + 3'd1;
                    state <= S_STORE_WRITE;
                end else begin
                    // Update bucket count
                    if (bucket_count[work_omega] < ENTRIES_PER_OMEGA - 1) begin
                        bucket_count[work_omega] <= bucket_count[work_omega] + 4'd1;
                    end
                    store_done <= 1'b1;
                    state <= S_IDLE;
                end
            end

            // ----------------------------------------------------------------
            // RECALL: Read entries from omega bucket
            // ----------------------------------------------------------------
            S_RECALL_READ: begin
                if (work_entry < bucket_count[work_omega]) begin
                    spram_addr <= calc_addr(work_omega, work_entry, work_word);
                    state <= S_RECALL_CHECK;
                end else begin
                    recall_done <= 1'b1;
                    state <= S_IDLE;
                end
            end

            S_RECALL_CHECK: begin
                // Read data from SPRAM (1 cycle latency)
                case (work_word)
                    3'd0: work_zeck[15:0] <= spram_data_out;
                    3'd1: work_zeck[31:16] <= spram_data_out;
                    3'd2: work_energy[15:0] <= spram_data_out;
                    3'd3: work_energy[31:16] <= spram_data_out;
                    3'd4: work_theta[15:0] <= spram_data_out;
                    3'd5: work_theta[31:16] <= spram_data_out;
                endcase

                if (work_word < 3'd5) begin
                    work_word <= work_word + 3'd1;
                    state <= S_RECALL_READ;
                end else begin
                    // Entry complete, output it
                    recall_found <= 1'b1;
                    recall_zeck <= work_zeck;
                    recall_energy <= work_energy;
                    recall_theta <= work_theta;
                    recall_done <= 1'b1;
                    state <= S_IDLE;
                end
            end

            // ----------------------------------------------------------------
            // SEARCH: Find similar patterns across omega range
            // ----------------------------------------------------------------
            S_SEARCH_INIT: begin
                work_entry <= 4'd0;
                work_word <= 3'd0;
                state <= S_SEARCH_READ;
            end

            S_SEARCH_READ: begin
                if (search_current_omega <= search_omega_max[OMEGA_WIDTH-1:0]) begin
                    if (work_entry < bucket_count[search_current_omega]) begin
                        spram_addr <= calc_addr(search_current_omega, work_entry, work_word);
                        state <= S_SEARCH_EVAL;
                    end else begin
                        // Move to next omega bucket
                        search_current_omega <= search_current_omega + 6'd1;
                        work_entry <= 4'd0;
                        work_word <= 3'd0;
                    end
                end else begin
                    // Search complete
                    search_done <= 1'b1;
                    state <= S_IDLE;
                end
            end

            S_SEARCH_EVAL: begin
                case (work_word)
                    3'd0: work_zeck[15:0] <= spram_data_out;
                    3'd1: work_zeck[31:16] <= spram_data_out;
                    3'd2: work_energy[15:0] <= spram_data_out;
                    3'd3: work_energy[31:16] <= spram_data_out;
                    3'd4: work_theta[15:0] <= spram_data_out;
                    3'd5: work_theta[31:16] <= spram_data_out;
                endcase

                if (work_word < 3'd5) begin
                    work_word <= work_word + 3'd1;
                    state <= S_SEARCH_READ;
                end else begin
                    // Evaluate this entry
                    search_count <= search_count + 4'd1;

                    // Track best by energy
                    if (work_energy > best_energy) begin
                        best_energy <= work_energy;
                        search_best_zeck <= work_zeck;
                        search_best_energy <= work_energy;
                    end

                    // Next entry
                    work_entry <= work_entry + 4'd1;
                    work_word <= 3'd0;
                    state <= S_SEARCH_READ;
                end
            end

            S_DONE: begin
                state <= S_IDLE;
            end
        endcase
    end
end

endmodule

`default_nettype wire
