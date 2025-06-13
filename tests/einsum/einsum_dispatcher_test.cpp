#include <gtest/gtest.h>

#include "fixtures/einsum.h"
#include "sdfg/codegen/code_generators/c_code_generator.h"

TEST(EinsumDispatcher, MatrixMatrixMultiplication) {
    auto sdfg_and_node = matrix_matrix_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, float **A, float **B, float **C)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float **_in1 = A;
        float **_in2 = B;
        float **_out;

        for (i = 0; i < I; i++)
        {
            for (k = 0; k < K; k++)
            {
                _out[i][k] = 0.0f;
                for (j = 0; j < J; j++)
                {
                    _out[i][k] = _out[i][k] + _in1[i][j] * _in2[j][k];
                }
            }
        }

        C = _out;
    }
)");
}

TEST(EinsumDispatcher, TensorContraction3D) {
    auto sdfg_and_node = tensor_contraction_3d();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, unsigned long long L, unsigned long long M, unsigned long long N, float ***A, "
              "float ***B, float ***C, float ***D)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float ***_in1 = A;
        float ***_in2 = B;
        float ***_in3 = C;
        float ***_out;

        for (i = 0; i < I; i++)
        {
            for (j = 0; j < J; j++)
            {
                for (k = 0; k < K; k++)
                {
                    _out[i][j][k] = 0.0f;
                    for (l = 0; l < L; l++)
                    {
                        for (m = 0; m < M; m++)
                        {
                            for (n = 0; n < N; n++)
                            {
                                _out[i][j][k] = _out[i][j][k] + _in1[l][j][m] * _in2[i][l][n] * _in3[n][m][k];
                            }
                        }
                    }
                }
            }
        }

        D = _out;
    }
)");
}

TEST(EinsumDispatcher, MatrixVectorMultiplication) {
    auto sdfg_and_node = matrix_vector_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, float **A, float *b, "
              "float *c)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float **_in1 = A;
        float *_in2 = b;
        float *_out;

        for (i = 0; i < I; i++)
        {
            _out[i] = 0.0f;
            for (j = 0; j < J; j++)
            {
                _out[i] = _out[i] + _in1[i][j] * _in2[j];
            }
        }

        c = _out;
    }
)");
}

TEST(EinsumDispatcher, DiagonalExtraction) {
    auto sdfg_and_node = diagonal_extraction();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float **A, float *b)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float **_in = A;
        float *_out;

        for (i = 0; i < I; i++)
        {
            _out[i] = 0.0f;
            _out[i] = _out[i] + _in[i][i];
        }

        b = _out;
    }
)");
}

TEST(EinsumDispatcher, MatrixTrace) {
    auto sdfg_and_node = matrix_trace();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float **A, float *b)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float **_in = A;
        float *_out;

        *_out = 0.0f;
        for (i = 0; i < I; i++)
        {
            *_out = *_out + _in[i][i];
        }

        b = _out;
    }
)");
}
