// =============================================================================
// ZECK_ENCODE - Integer to Zeckendorf Decomposition
// =============================================================================
// Greedy algorithm: take largest Fibonacci <= remaining, skip next (gap rule)
//
// Input: 32-bit integer value
// Output: N-bit shell occupancy (zeckbits)
//
// The bit at position i means "F_{i+2} is in the sum"
// (We start at F_2=1 because F_0=0, F_1=1 are edge cases)
// =============================================================================

module zeck_encode #(
    parameter N = 32,           // Number of shells
    parameter WIDTH = 32        // Input value width
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [WIDTH-1:0]  value_in,
    output reg  [N-1:0]      zeck_out,
    output reg               done
);

    // Fibonacci LUT (F_2 to F_{N+1})
    // F_2=1, F_3=2, F_4=3, F_5=5, F_6=8, ...
    reg [WIDTH-1:0] FIB [0:N-1];

    initial begin
        FIB[0]  = 32'd1;      // F_2
        FIB[1]  = 32'd2;      // F_3
        FIB[2]  = 32'd3;      // F_4
        FIB[3]  = 32'd5;      // F_5
        FIB[4]  = 32'd8;      // F_6
        FIB[5]  = 32'd13;     // F_7
        FIB[6]  = 32'd21;     // F_8
        FIB[7]  = 32'd34;     // F_9
        FIB[8]  = 32'd55;     // F_10
        FIB[9]  = 32'd89;     // F_11
        FIB[10] = 32'd144;    // F_12
        FIB[11] = 32'd233;    // F_13
        FIB[12] = 32'd377;    // F_14
        FIB[13] = 32'd610;    // F_15
        FIB[14] = 32'd987;    // F_16
        FIB[15] = 32'd1597;   // F_17
        FIB[16] = 32'd2584;   // F_18
        FIB[17] = 32'd4181;   // F_19
        FIB[18] = 32'd6765;   // F_20
        FIB[19] = 32'd10946;  // F_21
        FIB[20] = 32'd17711;  // F_22
        FIB[21] = 32'd28657;  // F_23
        FIB[22] = 32'd46368;  // F_24
        FIB[23] = 32'd75025;  // F_25
        FIB[24] = 32'd121393; // F_26
        FIB[25] = 32'd196418; // F_27
        FIB[26] = 32'd317811; // F_28
        FIB[27] = 32'd514229; // F_29
        FIB[28] = 32'd832040; // F_30
        FIB[29] = 32'd1346269;// F_31
        FIB[30] = 32'd2178309;// F_32
        FIB[31] = 32'd3524578;// F_33
    end

    reg [WIDTH-1:0] remaining;
    reg [5:0] idx;
    reg [N-1:0] result;
    reg [1:0] state;

    localparam IDLE = 2'b00;
    localparam SCAN = 2'b01;
    localparam DONE = 2'b10;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            zeck_out <= {N{1'b0}};
            remaining <= 0;
            idx <= 0;
            result <= {N{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        remaining <= value_in;
                        idx <= N - 1;  // Start from largest Fibonacci
                        result <= {N{1'b0}};
                        state <= SCAN;
                    end
                end

                SCAN: begin
                    if (remaining == 0) begin
                        state <= DONE;
                    end else if (idx == 0 && FIB[0] > remaining) begin
                        // Can't decompose further
                        state <= DONE;
                    end else begin
                        if (FIB[idx] <= remaining) begin
                            // Take this Fibonacci
                            result[idx] <= 1'b1;
                            remaining <= remaining - FIB[idx];
                            // Skip next (Zeckendorf gap rule)
                            idx <= (idx >= 2) ? idx - 2 : 6'd0;
                        end else begin
                            idx <= (idx > 0) ? idx - 1 : 6'd0;
                        end

                        if (idx == 0) begin
                            state <= DONE;
                        end
                    end
                end

                DONE: begin
                    zeck_out <= result;
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
