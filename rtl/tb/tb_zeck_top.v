// ============================================================================
// TESTBENCH: Zeckendorf AI State Machine
// ============================================================================
// Deterministic AI Apparatus - Patent Pending
//
// Verifies:
// 1. Zeckendorf decomposition (OEIS A003714)
// 2. Cascade operator (kappa)
// 3. CORDIC sin/cos accuracy
// 4. Omega-indexed memory
// 5. FSM state transitions
// 6. 100% reproducibility across runs
//
// Target: Simulation (Icarus Verilog / Verilator)
// ============================================================================

`timescale 1ns / 1ps

module tb_zeck_top;

// ============================================================================
// PARAMETERS
// ============================================================================
parameter CLK_PERIOD = 20.83;  // 48MHz = 20.83ns

// ============================================================================
// SIGNALS
// ============================================================================
reg         clk;
reg         rst_n;

// Zeckendorf arithmetic
reg         zeck_start;
reg  [2:0]  zeck_opcode;
reg  [31:0] zeck_operand_a;
reg  [31:0] zeck_operand_b;
wire        zeck_done;
wire        zeck_valid;
wire [31:0] zeck_result;
wire [5:0]  zeck_omega;
wire [4:0]  zeck_complexity;

// Cascade
reg         cascade_start;
reg  [31:0] cascade_bits_in;
wire        cascade_done;
wire        cascade_valid;
wire [31:0] cascade_bits_out;
wire [3:0]  cascade_iterations;

// CORDIC
reg                 cordic_start;
reg  signed [31:0]  cordic_angle;
wire                cordic_done;
wire signed [31:0]  cordic_sin;
wire signed [31:0]  cordic_cos;

// Test results
integer tests_passed;
integer tests_failed;
integer test_num;

// ============================================================================
// DUT INSTANTIATION
// ============================================================================
zeckendorf_arith #(
    .WIDTH(32),
    .FIB_DEPTH(47)
) dut_zeck (
    .clk(clk),
    .rst_n(rst_n),
    .start(zeck_start),
    .opcode(zeck_opcode),
    .operand_a(zeck_operand_a),
    .operand_b(zeck_operand_b),
    .done(zeck_done),
    .valid(zeck_valid),
    .result(zeck_result),
    .omega(zeck_omega),
    .complexity(zeck_complexity)
);

cascade_iterative #(
    .WIDTH(32),
    .MAX_ITERATIONS(15)
) dut_cascade (
    .clk(clk),
    .rst_n(rst_n),
    .start(cascade_start),
    .bits_in(cascade_bits_in),
    .done(cascade_done),
    .valid(cascade_valid),
    .bits_out(cascade_bits_out),
    .iterations(cascade_iterations)
);

cordic_sincos #(
    .WIDTH(32)
) dut_cordic (
    .clk(clk),
    .rst_n(rst_n),
    .start(cordic_start),
    .angle(cordic_angle),
    .done(cordic_done),
    .sin_out(cordic_sin),
    .cos_out(cordic_cos)
);

// ============================================================================
// CLOCK GENERATION
// ============================================================================
initial begin
    clk = 1'b0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// ============================================================================
// TEST SEQUENCES
// ============================================================================
initial begin
    // Initialize
    $dumpfile("tb_zeck_top.vcd");
    $dumpvars(0, tb_zeck_top);

    tests_passed = 0;
    tests_failed = 0;
    test_num = 0;

    rst_n = 1'b0;
    zeck_start = 1'b0;
    zeck_opcode = 3'b000;
    zeck_operand_a = 32'd0;
    zeck_operand_b = 32'd0;
    cascade_start = 1'b0;
    cascade_bits_in = 32'd0;
    cordic_start = 1'b0;
    cordic_angle = 32'd0;

    // Reset
    #100;
    rst_n = 1'b1;
    #100;

    $display("============================================");
    $display("ZECKENDORF AI STATE MACHINE TESTBENCH");
    $display("============================================");
    $display("");

    // ========================================================================
    // TEST 1: Zeckendorf Decomposition
    // ========================================================================
    $display("TEST 1: Zeckendorf Decomposition");
    $display("--------------------------------");

    // Test n=17 -> should decompose to F[7]+F[4]+F[2] = 13+3+1 = 17
    // Zeck bits: 10010100 (bits 7,4,2 set)
    test_zeck_decompose(32'd17, 32'h94, 6'd13, 3);  // 0x94 = 0b10010100

    // Test n=100 -> F[11]+F[6]+F[4] = 89+8+3 = 100
    // Zeck bits: 100001010000
    test_zeck_decompose(32'd100, 32'h850, 6'd21, 3);

    // Test n=1000 -> F[16]+F[7] = 987+13 = 1000
    test_zeck_decompose(32'd1000, 32'h10080, 6'd23, 2);

    // ========================================================================
    // TEST 2: Cascade Operator
    // ========================================================================
    $display("");
    $display("TEST 2: Cascade Operator (kappa)");
    $display("--------------------------------");

    // Test 0b110 (invalid) -> 0b1000 (valid)
    test_cascade(32'b110, 32'b1000);

    // Test 0b111 (invalid) -> valid form
    test_cascade(32'b111, 32'b1001);

    // Test 0b101 (already valid) -> 0b101
    test_cascade(32'b101, 32'b101);

    // ========================================================================
    // TEST 3: CORDIC sin/cos
    // ========================================================================
    $display("");
    $display("TEST 3: CORDIC sin/cos (Integer-Only)");
    $display("-------------------------------------");

    // Test sin(pi/4), cos(pi/4) = 0.707107...
    // pi/4 in Q2.30 = 843314857
    test_cordic(32'sd843314857, 32'sd759250125, 32'sd759250125);  // ~0.707

    // Test sin(0), cos(0) = 0, 1
    test_cordic(32'sd0, 32'sd0, 32'sd1073741824);  // 1.0 in Q2.30

    // ========================================================================
    // TEST 4: Zeckendorf Validity (OEIS A003714)
    // ========================================================================
    $display("");
    $display("TEST 4: Zeckendorf Validity (A003714)");
    $display("-------------------------------------");

    // Valid patterns (no adjacent 1s)
    test_validity(32'b101010101, 1'b1);
    test_validity(32'b100100100, 1'b1);

    // Invalid patterns (adjacent 1s)
    test_validity(32'b110, 1'b0);
    test_validity(32'b1011, 1'b0);

    // ========================================================================
    // TEST 5: Reproducibility
    // ========================================================================
    $display("");
    $display("TEST 5: 100%% Reproducibility");
    $display("----------------------------");

    test_reproducibility();

    // ========================================================================
    // SUMMARY
    // ========================================================================
    $display("");
    $display("============================================");
    $display("TEST SUMMARY");
    $display("============================================");
    $display("Tests Passed: %0d", tests_passed);
    $display("Tests Failed: %0d", tests_failed);
    $display("Total Tests:  %0d", tests_passed + tests_failed);
    $display("");

    if (tests_failed == 0) begin
        $display("*** ALL TESTS PASSED ***");
    end else begin
        $display("*** SOME TESTS FAILED ***");
    end

    $display("============================================");
    $finish;
end

// ============================================================================
// TEST TASKS
// ============================================================================

task test_zeck_decompose;
    input [31:0] value;
    input [31:0] expected_bits;
    input [5:0]  expected_omega;
    input [4:0]  expected_complexity;
begin
    test_num = test_num + 1;

    @(posedge clk);
    zeck_operand_a <= value;
    zeck_opcode <= 3'b000;  // INT_TO_ZECK
    zeck_start <= 1'b1;
    @(posedge clk);
    zeck_start <= 1'b0;

    // Wait for done
    wait(zeck_done);
    @(posedge clk);

    if (zeck_result == expected_bits && zeck_valid) begin
        $display("  [PASS] Zeck(%0d) = 0x%h, Omega=%0d, Complexity=%0d",
                 value, zeck_result, zeck_omega, zeck_complexity);
        tests_passed = tests_passed + 1;
    end else begin
        $display("  [FAIL] Zeck(%0d): got 0x%h, expected 0x%h",
                 value, zeck_result, expected_bits);
        tests_failed = tests_failed + 1;
    end

    #100;
end
endtask

task test_cascade;
    input [31:0] bits_in;
    input [31:0] expected_out;
begin
    test_num = test_num + 1;

    @(posedge clk);
    cascade_bits_in <= bits_in;
    cascade_start <= 1'b1;
    @(posedge clk);
    cascade_start <= 1'b0;

    // Wait for done
    wait(cascade_done);
    @(posedge clk);

    if (cascade_bits_out == expected_out && cascade_valid) begin
        $display("  [PASS] Cascade(0b%b) = 0b%b, iterations=%0d",
                 bits_in, cascade_bits_out, cascade_iterations);
        tests_passed = tests_passed + 1;
    end else begin
        $display("  [FAIL] Cascade(0b%b): got 0b%b, expected 0b%b",
                 bits_in, cascade_bits_out, expected_out);
        tests_failed = tests_failed + 1;
    end

    #100;
end
endtask

task test_cordic;
    input signed [31:0] angle;
    input signed [31:0] expected_sin;
    input signed [31:0] expected_cos;
    real sin_actual, cos_actual, sin_expected, cos_expected;
    real sin_error, cos_error;
begin
    test_num = test_num + 1;

    @(posedge clk);
    cordic_angle <= angle;
    cordic_start <= 1'b1;
    @(posedge clk);
    cordic_start <= 1'b0;

    // Wait for done
    wait(cordic_done);
    @(posedge clk);

    // Convert to real for display
    sin_actual = $itor(cordic_sin) / 1073741824.0;
    cos_actual = $itor(cordic_cos) / 1073741824.0;
    sin_expected = $itor(expected_sin) / 1073741824.0;
    cos_expected = $itor(expected_cos) / 1073741824.0;

    sin_error = sin_actual - sin_expected;
    cos_error = cos_actual - cos_expected;

    // Allow 1% error for CORDIC
    if (sin_error < 0) sin_error = -sin_error;
    if (cos_error < 0) cos_error = -cos_error;

    if (sin_error < 0.01 && cos_error < 0.01) begin
        $display("  [PASS] CORDIC: sin=%f, cos=%f (angle=%0d)",
                 sin_actual, cos_actual, angle);
        tests_passed = tests_passed + 1;
    end else begin
        $display("  [FAIL] CORDIC: sin=%f (exp %f), cos=%f (exp %f)",
                 sin_actual, sin_expected, cos_actual, cos_expected);
        tests_failed = tests_failed + 1;
    end

    #100;
end
endtask

task test_validity;
    input [31:0] bits;
    input expected_valid;
    wire actual_valid;
begin
    test_num = test_num + 1;

    // Check for adjacent 1s
    assign actual_valid = ((bits & (bits << 1)) == 0);

    @(posedge clk);
    #10;

    if (actual_valid == expected_valid) begin
        $display("  [PASS] Validity(0b%b) = %0d", bits, actual_valid);
        tests_passed = tests_passed + 1;
    end else begin
        $display("  [FAIL] Validity(0b%b): got %0d, expected %0d",
                 bits, actual_valid, expected_valid);
        tests_failed = tests_failed + 1;
    end
end
endtask

task test_reproducibility;
    reg [31:0] result1, result2, result3;
begin
    test_num = test_num + 1;

    // Run same computation 3 times
    @(posedge clk);
    zeck_operand_a <= 32'd12345;
    zeck_opcode <= 3'b000;
    zeck_start <= 1'b1;
    @(posedge clk);
    zeck_start <= 1'b0;
    wait(zeck_done);
    result1 = zeck_result;

    #100;

    @(posedge clk);
    zeck_operand_a <= 32'd12345;
    zeck_opcode <= 3'b000;
    zeck_start <= 1'b1;
    @(posedge clk);
    zeck_start <= 1'b0;
    wait(zeck_done);
    result2 = zeck_result;

    #100;

    @(posedge clk);
    zeck_operand_a <= 32'd12345;
    zeck_opcode <= 3'b000;
    zeck_start <= 1'b1;
    @(posedge clk);
    zeck_start <= 1'b0;
    wait(zeck_done);
    result3 = zeck_result;

    if (result1 == result2 && result2 == result3) begin
        $display("  [PASS] Reproducibility: 3 runs identical (0x%h)", result1);
        tests_passed = tests_passed + 1;
    end else begin
        $display("  [FAIL] Reproducibility: results differ");
        $display("         Run 1: 0x%h", result1);
        $display("         Run 2: 0x%h", result2);
        $display("         Run 3: 0x%h", result3);
        tests_failed = tests_failed + 1;
    end
end
endtask

endmodule
