#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_matrix_mult() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    builder.add_container("J", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 std::vector<std::string>, std::vector<std::vector<std::string>>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in1", "_in2"}, false, DebugInfo(),
            {{symbolic::symbol("i"), symbolic::symbol("I")},
             {symbolic::symbol("j"), symbolic::symbol("J")},
             {symbolic::symbol("k"), symbolic::symbol("K")}},
            {"i", "k"}, {{"i", "j"}, {"j", "k"}});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", C, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> tensor_contraction_3d() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    builder.add_container("J", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("L", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    types::Pointer desc3(*desc2.clone());
    builder.add_container("A", desc3, true);
    builder.add_container("B", desc3, true);
    builder.add_container("C", desc3, true);
    builder.add_container("D", desc3, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 std::vector<std::string>, std::vector<std::vector<std::string>>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in1", "_in2", "_in3"}, false,
            DebugInfo(),
            {{symbolic::symbol("i"), symbolic::symbol("I")},
             {symbolic::symbol("j"), symbolic::symbol("J")},
             {symbolic::symbol("k"), symbolic::symbol("K")},
             {symbolic::symbol("l"), symbolic::symbol("L")},
             {symbolic::symbol("m"), symbolic::symbol("M")},
             {symbolic::symbol("n"), symbolic::symbol("N")}},
            {"i", "j", "k"}, {{"l", "j", "m"}, {"i", "l", "n"}, {"n", "m", "k"}});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, C, "void", libnode, "_in3", {});
    builder.add_memlet(block, libnode, "_out", D, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_vector_mult() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    builder.add_container("J", sym_desc, true);
    
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 std::vector<std::string>, std::vector<std::vector<std::string>>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in1", "_in2"}, false, DebugInfo(),
            {{symbolic::symbol("i"), symbolic::symbol("I")},
             {symbolic::symbol("j"), symbolic::symbol("J")}},
            {"i"}, {{"i", "j"}, {"j"}});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, b, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", c, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> diagonal_extraction() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 std::vector<std::string>, std::vector<std::vector<std::string>>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in"}, false, DebugInfo(),
            {{symbolic::symbol("i"), symbolic::symbol("I")}}, {"i"}, {{"i", "i"}});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", b, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_trace() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 std::vector<std::string>, std::vector<std::vector<std::string>>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in"}, false, DebugInfo(),
            {{symbolic::symbol("i"), symbolic::symbol("I")}}, {}, {{"i", "i"}});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", b, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}
