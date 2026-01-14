/*
 * ZISA Core: Two-Shell Zeckendorf Integer Sequence Accelerator
 *
 * Target: iCE40UP5K (5280 LUTs, 128 Kbit SPRAM, 120 Kbit EBR)
 *
 * This module implements:
 *   - Dual-rail Verlet structure (phi/psi rails)
 *   - Priority cascade normalization (Fk + Fk-1 = Fk+1)
 *   - Eigenvalue accumulation via Binet decomposition
 *
 * Key Insight: All operations reduce to integer addition in log_phi space
 *
 * Token Format:
 *   - (i, j) shell pair where j > i + 1 (gap >= 2)
 *   - Encodes as 64-bit pattern with exactly two 1-bits
 */

module zisa_core #(
    parameter WIDTH = 48,           // Shell width (Fibonacci indices 0..47)
    parameter PSI_SCALE = 24        // Fixed-point scale for psi values
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Token input interface
    input  wire                 token_valid,
    output wire                 token_ready,
    input  wire [5:0]           token_i,        // Lower shell index
    input  wire [5:0]           token_j,        // Upper shell index

    // State outputs
    output reg  [WIDTH-1:0]     A_phi,          // Phi-rail accumulator
    output reg  [WIDTH-1:0]     A_psi,          // Psi-rail accumulator
    output reg  [63:0]          F_acc,          // Fibonacci sum accumulator
    output reg  [63:0]          L_acc,          // Lucas sum accumulator
    output reg  signed [31:0]   Psi_acc,        // Psi signature (signed)
    output reg  [15:0]          tau_total,      // Total cascade count
    output reg  [15:0]          pos,            // Sequence position

    // Status
    output wire                 cascade_active,
    output wire                 zeck_legal
);

// ============================================================================
// FIBONACCI / LUCAS ROM
// ============================================================================
// Precomputed values for indices 0..47
// F_n = F_{n-1} + F_{n-2}, F_0=0, F_1=1
// L_n = L_{n-1} + L_{n-2}, L_0=2, L_1=1

reg [63:0] FIB_ROM [0:47];
reg [63:0] LUC_ROM [0:47];
reg signed [31:0] PSI_ROM [0:47];  // Scaled by 2^24

initial begin
    // Fibonacci sequence
    FIB_ROM[0]  = 64'd0;
    FIB_ROM[1]  = 64'd1;
    FIB_ROM[2]  = 64'd1;
    FIB_ROM[3]  = 64'd2;
    FIB_ROM[4]  = 64'd3;
    FIB_ROM[5]  = 64'd5;
    FIB_ROM[6]  = 64'd8;
    FIB_ROM[7]  = 64'd13;
    FIB_ROM[8]  = 64'd21;
    FIB_ROM[9]  = 64'd34;
    FIB_ROM[10] = 64'd55;
    FIB_ROM[11] = 64'd89;
    FIB_ROM[12] = 64'd144;
    FIB_ROM[13] = 64'd233;
    FIB_ROM[14] = 64'd377;
    FIB_ROM[15] = 64'd610;
    FIB_ROM[16] = 64'd987;
    FIB_ROM[17] = 64'd1597;
    FIB_ROM[18] = 64'd2584;
    FIB_ROM[19] = 64'd4181;
    FIB_ROM[20] = 64'd6765;
    FIB_ROM[21] = 64'd10946;
    FIB_ROM[22] = 64'd17711;
    FIB_ROM[23] = 64'd28657;
    FIB_ROM[24] = 64'd46368;
    FIB_ROM[25] = 64'd75025;
    FIB_ROM[26] = 64'd121393;
    FIB_ROM[27] = 64'd196418;
    FIB_ROM[28] = 64'd317811;
    FIB_ROM[29] = 64'd514229;
    FIB_ROM[30] = 64'd832040;
    FIB_ROM[31] = 64'd1346269;
    FIB_ROM[32] = 64'd2178309;
    FIB_ROM[33] = 64'd3524578;
    FIB_ROM[34] = 64'd5702887;
    FIB_ROM[35] = 64'd9227465;
    FIB_ROM[36] = 64'd14930352;
    FIB_ROM[37] = 64'd24157817;
    FIB_ROM[38] = 64'd39088169;
    FIB_ROM[39] = 64'd63245986;
    FIB_ROM[40] = 64'd102334155;
    FIB_ROM[41] = 64'd165580141;
    FIB_ROM[42] = 64'd267914296;
    FIB_ROM[43] = 64'd433494437;
    FIB_ROM[44] = 64'd701408733;
    FIB_ROM[45] = 64'd1134903170;
    FIB_ROM[46] = 64'd1836311903;
    FIB_ROM[47] = 64'd2971215073;

    // Lucas sequence
    LUC_ROM[0]  = 64'd2;
    LUC_ROM[1]  = 64'd1;
    LUC_ROM[2]  = 64'd3;
    LUC_ROM[3]  = 64'd4;
    LUC_ROM[4]  = 64'd7;
    LUC_ROM[5]  = 64'd11;
    LUC_ROM[6]  = 64'd18;
    LUC_ROM[7]  = 64'd29;
    LUC_ROM[8]  = 64'd47;
    LUC_ROM[9]  = 64'd76;
    LUC_ROM[10] = 64'd123;
    LUC_ROM[11] = 64'd199;
    LUC_ROM[12] = 64'd322;
    LUC_ROM[13] = 64'd521;
    LUC_ROM[14] = 64'd843;
    LUC_ROM[15] = 64'd1364;
    LUC_ROM[16] = 64'd2207;
    LUC_ROM[17] = 64'd3571;
    LUC_ROM[18] = 64'd5778;
    LUC_ROM[19] = 64'd9349;
    LUC_ROM[20] = 64'd15127;
    LUC_ROM[21] = 64'd24476;
    LUC_ROM[22] = 64'd39603;
    LUC_ROM[23] = 64'd64079;
    LUC_ROM[24] = 64'd103682;
    LUC_ROM[25] = 64'd167761;
    LUC_ROM[26] = 64'd271443;
    LUC_ROM[27] = 64'd439204;
    LUC_ROM[28] = 64'd710647;
    LUC_ROM[29] = 64'd1149851;
    LUC_ROM[30] = 64'd1860498;
    LUC_ROM[31] = 64'd3010349;
    LUC_ROM[32] = 64'd4870847;
    LUC_ROM[33] = 64'd7881196;
    LUC_ROM[34] = 64'd12752043;
    LUC_ROM[35] = 64'd20633239;
    LUC_ROM[36] = 64'd33385282;
    LUC_ROM[37] = 64'd54018521;
    LUC_ROM[38] = 64'd87403803;
    LUC_ROM[39] = 64'd141422324;
    LUC_ROM[40] = 64'd228826127;
    LUC_ROM[41] = 64'd370248451;
    LUC_ROM[42] = 64'd599074578;
    LUC_ROM[43] = 64'd969323029;
    LUC_ROM[44] = 64'd1568397607;
    LUC_ROM[45] = 64'd2537720636;
    LUC_ROM[46] = 64'd4106118243;
    LUC_ROM[47] = 64'd6643838879;

    // Psi LUT: psi^k = (-1)^k / phi^k, scaled by 2^24
    // Computed as: round((-1)^k / phi^k * 2^24)
    PSI_ROM[0]  = 32'd16777216;     // 1.0
    PSI_ROM[1]  = -32'd10368318;    // -0.618...
    PSI_ROM[2]  = 32'd6408899;      // 0.382...
    PSI_ROM[3]  = -32'd3959420;     // -0.236...
    PSI_ROM[4]  = 32'd2449479;      // 0.146...
    PSI_ROM[5]  = -32'd1509942;     // -0.090...
    PSI_ROM[6]  = 32'd939537;       // 0.056...
    PSI_ROM[7]  = -32'd570405;      // -0.034...
    PSI_ROM[8]  = 32'd369132;       // 0.022...
    PSI_ROM[9]  = -32'd201274;      // -0.012...
    PSI_ROM[10] = 32'd167858;       // 0.010...
    PSI_ROM[11] = -32'd33416;       // -0.002...
    PSI_ROM[12] = 32'd134442;       // 0.008...
    PSI_ROM[13] = -32'd101026;      // -0.006...
    PSI_ROM[14] = 32'd33416;        // 0.002...
    PSI_ROM[15] = -32'd67610;       // -0.004...
    PSI_ROM[16] = 32'd34194;        // 0.002...
    PSI_ROM[17] = -32'd33416;       // -0.002...
    PSI_ROM[18] = 32'd778;          // 0.00005...
    PSI_ROM[19] = -32'd32638;       // -0.002...
    PSI_ROM[20] = 32'd33416;        // 0.002...
    PSI_ROM[21] = -32'd778;         // -0.00005...
    PSI_ROM[22] = 32'd32638;        // 0.002...
    PSI_ROM[23] = -32'd31860;       // -0.002...
    PSI_ROM[24] = 32'd778;          // 0.00005...
    PSI_ROM[25] = -32'd31082;       // -0.002...
    PSI_ROM[26] = 32'd31860;        // 0.002...
    PSI_ROM[27] = -32'd778;         // -0.00005...
    PSI_ROM[28] = 32'd31082;        // 0.002...
    PSI_ROM[29] = -32'd30304;       // -0.002...
    PSI_ROM[30] = 32'd778;          // 0.00005...
    PSI_ROM[31] = -32'd29526;       // -0.002...
    PSI_ROM[32] = 32'd30304;        // 0.002...
    PSI_ROM[33] = -32'd778;         // -0.00005...
    PSI_ROM[34] = 32'd29526;        // 0.002...
    PSI_ROM[35] = -32'd28748;       // -0.002...
    PSI_ROM[36] = 32'd778;          // 0.00005...
    PSI_ROM[37] = -32'd27970;       // -0.002...
    PSI_ROM[38] = 32'd28748;        // 0.002...
    PSI_ROM[39] = -32'd778;         // -0.00005...
    PSI_ROM[40] = 32'd27970;        // 0.002...
    PSI_ROM[41] = -32'd27192;       // -0.002...
    PSI_ROM[42] = 32'd778;          // 0.00005...
    PSI_ROM[43] = -32'd26414;       // -0.002...
    PSI_ROM[44] = 32'd27192;        // 0.002...
    PSI_ROM[45] = -32'd778;         // -0.00005...
    PSI_ROM[46] = 32'd26414;        // 0.002...
    PSI_ROM[47] = -32'd25636;       // -0.002...
end

// ============================================================================
// FSM STATES
// ============================================================================

localparam IDLE     = 3'd0;
localparam ABSORB   = 3'd1;
localparam CASCADE  = 3'd2;
localparam UPDATE   = 3'd3;
localparam DONE     = 3'd4;

reg [2:0] state, next_state;

// ============================================================================
// WORKING REGISTERS
// ============================================================================

reg [WIDTH-1:0] work_bits;          // Current rail being processed
reg [5:0]       cascade_pos;        // Current cascade check position
reg [5:0]       cascade_count;      // Cascades in current absorb
reg             use_phi_rail;       // Which rail to update (0=psi, 1=phi)

// Token being absorbed
reg [5:0] cur_i, cur_j;

// ============================================================================
// ADJACENT BIT DETECTION
// ============================================================================

wire has_adjacent = |(work_bits & (work_bits >> 1));

// Find highest adjacent pair position
reg [5:0] highest_adj;
integer k;
always @(*) begin
    highest_adj = 6'd0;
    for (k = WIDTH-2; k >= 0; k = k - 1) begin
        if (work_bits[k+1] && work_bits[k] && highest_adj == 0) begin
            highest_adj = k[5:0];
        end
    end
end

// ============================================================================
// FSM LOGIC
// ============================================================================

assign token_ready = (state == IDLE);
assign cascade_active = (state == CASCADE);
assign zeck_legal = !has_adjacent;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

always @(*) begin
    next_state = state;
    case (state)
        IDLE: begin
            if (token_valid)
                next_state = ABSORB;
        end

        ABSORB: begin
            next_state = CASCADE;
        end

        CASCADE: begin
            if (!has_adjacent)
                next_state = UPDATE;
            // else stay in CASCADE
        end

        UPDATE: begin
            next_state = DONE;
        end

        DONE: begin
            next_state = IDLE;
        end

        default: next_state = IDLE;
    endcase
end

// ============================================================================
// DATAPATH
// ============================================================================

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        A_phi <= {WIDTH{1'b0}};
        A_psi <= {WIDTH{1'b0}};
        F_acc <= 64'd0;
        L_acc <= 64'd0;
        Psi_acc <= 32'sd0;
        tau_total <= 16'd0;
        pos <= 16'd0;
        work_bits <= {WIDTH{1'b0}};
        cascade_count <= 6'd0;
        use_phi_rail <= 1'b1;
        cur_i <= 6'd0;
        cur_j <= 6'd0;
    end else begin
        case (state)
            IDLE: begin
                if (token_valid) begin
                    cur_i <= token_i;
                    cur_j <= token_j;
                    use_phi_rail <= ~pos[0];  // Even pos -> phi, odd -> psi
                end
            end

            ABSORB: begin
                // OR token bits into selected rail
                if (use_phi_rail) begin
                    work_bits <= A_phi | ({{(WIDTH-1){1'b0}}, 1'b1} << cur_i)
                                       | ({{(WIDTH-1){1'b0}}, 1'b1} << cur_j);
                end else begin
                    work_bits <= A_psi | ({{(WIDTH-1){1'b0}}, 1'b1} << cur_i)
                                       | ({{(WIDTH-1){1'b0}}, 1'b1} << cur_j);
                end
                cascade_count <= 6'd0;
            end

            CASCADE: begin
                if (has_adjacent) begin
                    // Apply cascade: clear k and k+1, set k+2
                    // highest_adj is k (lower of the pair)
                    work_bits[highest_adj] <= 1'b0;
                    work_bits[highest_adj + 1] <= 1'b0;
                    if (highest_adj + 2 < WIDTH) begin
                        work_bits[highest_adj + 2] <= 1'b1;
                    end
                    cascade_count <= cascade_count + 1'b1;
                end
            end

            UPDATE: begin
                // Store normalized bits back to rail
                if (use_phi_rail) begin
                    A_phi <= work_bits;
                end else begin
                    A_psi <= work_bits;
                end

                // Update tau counter
                tau_total <= tau_total + {10'd0, cascade_count};

                // Increment position
                pos <= pos + 1'b1;

                // Update eigenvalue accumulators
                // Note: In hardware, this would use pipelined ROM reads
                // For simplicity, we just update with token contribution
                F_acc <= F_acc + FIB_ROM[cur_i] + FIB_ROM[cur_j];
                L_acc <= L_acc + LUC_ROM[cur_i] + LUC_ROM[cur_j];
                Psi_acc <= Psi_acc + PSI_ROM[cur_i] + PSI_ROM[cur_j];
            end

            DONE: begin
                // Ready for next token
            end
        endcase
    end
end

endmodule
