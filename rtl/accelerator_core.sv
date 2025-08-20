`timescale 1ns / 1ps
module accelerator_core (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,      
    input  logic [255:0] data_in,    // packed 8x32 words: each word [31:16]=data[2*i+1], [15:0]=data[2*i]
    input  logic [255:0] wei_in,     // same packing for weights
    input  logic [31:0]  bias_in,    
    output logic         busy,
    output logic         done,       // 1-cycle pulse when result ready
    output logic [31:0]  result      
);

    localparam int NUM_TAPS = 16;

    // Latched copies of inputs (sampled on START)
    logic signed [15:0] data_reg [0:NUM_TAPS-1];
    logic signed [15:0] wei_reg  [0:NUM_TAPS-1];
    logic signed [31:0] bias_reg;

    // Control & pipeline
    typedef enum logic [1:0] {S_IDLE = 2'd0, S_EXEC = 2'd1, S_DONE = 2'd2} state_t;
    state_t state;

    logic [4:0] issue_idx;   
    logic [4:0] add_count;   

    // pipeline registers
    logic signed [31:0] prod_reg;   
    logic               prod_valid; 

    logic signed [31:0] acc;        

    logic signed [31:0] mul_comb;

    // Latch packed inputs into data_reg/wei_reg on START
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_TAPS; i = i + 1) begin
                data_reg[i] <= 16'sd0;
                wei_reg[i]  <= 16'sd0;
            end
            bias_reg <= 32'sd0;
        end else begin
            if (state == S_IDLE && start) begin
                // Unpack 8 32-bit words into 16 16-bit signed elements
                for (i = 0; i < 8; i = i + 1) begin
                    logic [31:0] dword = data_in[i*32 +: 32];
                    logic [31:0] wword = wei_in[i*32 +: 32];
                    data_reg[2*i]   <= $signed(dword[15:0]);
                    data_reg[2*i+1] <= $signed(dword[31:16]);
                    wei_reg[2*i]    <= $signed(wword[15:0]);
                    wei_reg[2*i+1]  <= $signed(wword[31:16]);
                end
                bias_reg <= $signed(bias_in);
            end
        end
    end

    // Combinational multiply based on the current issue_idx and latched arrays
    always_comb begin
        if (issue_idx < NUM_TAPS) begin
            mul_comb = $signed(data_reg[issue_idx]) * $signed(wei_reg[issue_idx]);
        end else begin
            mul_comb = 32'sd0;
        end
    end

    // Main FSM and datapath
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            issue_idx <= 5'd0;
            add_count <= 5'd0;
            prod_reg <= 32'sd0;
            prod_valid <= 1'b0;
            acc <= 32'sd0;
            busy <= 1'b0;
            done <= 1'b0;
            result <= 32'd0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    prod_valid <= 1'b0;
                    issue_idx <= 5'd0;
                    add_count <= 5'd0;
                    acc <= 32'sd0;

                    if (start) begin
                        acc <= bias_reg;
                        busy <= 1'b1;
                        prod_reg <= 32'sd0;
                        prod_valid <= 1'b0;
                        issue_idx <= 5'd0;
                        add_count <= 5'd0;
                        state <= S_EXEC;
                    end
                end

                S_EXEC: begin
                    busy <= 1'b1;

                    if (prod_valid) begin
                        acc <= acc + (prod_reg >>> 15);
                        add_count <= add_count + 1;
                    end

                    prod_reg <= mul_comb;
                    prod_valid <= 1'b1;

                    if (issue_idx < NUM_TAPS - 1) begin
                        issue_idx <= issue_idx + 1;
                    end else begin
                        issue_idx <= issue_idx; 
                    end

                    if (add_count == NUM_TAPS) begin
                        prod_valid <= 1'b0;
                        busy <= 1'b0;
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    logic signed [31:0] final_val = acc;

                    if (final_val > 32'sd32767) begin
                        result <= {16'h0000, 16'h7FFF};
                    end else if (final_val < -32'sd32768) begin
                        result <= {16'h0000, 16'h8000};
                    end else begin
                        result <= {16'h0000, final_val[15:0]};
                    end

                    done <= 1'b1; 
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
