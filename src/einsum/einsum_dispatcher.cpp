#include "sdfg/einsum/einsum_dispatcher.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace einsum {

std::string EinsumDispatcher::containerSubset(const std::string& container,
                                              const std::vector<std::string>& indices) const {
    std::stringstream stream;

    if (indices.size() > 0) {
        stream << container;
        for (auto& index : indices) {
            stream << "[" << index << "]";
        }
    } else {
        stream << "*" << container;
    }

    return stream.str();
}

EinsumDispatcher::EinsumDispatcher(codegen::LanguageExtension& language_extension,
                                   const Function& function,
                                   const data_flow::DataFlowGraph& data_flow_graph,
                                   const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void EinsumDispatcher::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(this->node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = this->function_.type(src.data());

        auto& conn_name = iedge.dst_conn();
        auto& conn_type = types::infer_type(function_, src_type, iedge.subset());

        stream << this->language_extension_.declaration(conn_name, conn_type);

        stream << " = " << src.data()
               << this->language_extension_.subset(function_, src_type, iedge.subset()) << ";"
               << std::endl;
    }
    for (auto& oedge : this->data_flow_graph_.out_edges(this->node_)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& dst_type = this->function_.type(dst.data());

        auto& conn_name = oedge.src_conn();
        auto& conn_type = types::infer_type(function_, dst_type, oedge.subset());
        stream << this->language_extension_.declaration(conn_name, conn_type) << ";" << std::endl;
    }

    stream << std::endl;

    const EinsumNode* einsum_node = dynamic_cast<const EinsumNode*>(&this->node_);
    size_t nested = 0;

    // Create outer maps as for loops
    for (std::string out_index : einsum_node->out_indices()) {
        for (const std::pair<symbolic::Symbol, symbolic::Expression>& map : einsum_node->maps()) {
            if (map.first->get_name() != out_index) continue;
            stream << "for (" << map.first->get_name() << " = 0; " << map.first->get_name() << " < "
                   << this->language_extension_.expression(map.second) << "; "
                   << map.first->get_name() << "++)" << std::endl
                   << "{" << std::endl;
            stream.setIndent(stream.indent() + 4);
            ++nested;
        }
    }

    // Set out container entry to 0
    stream << this->containerSubset(einsum_node->output(0), einsum_node->out_indices()) << " = ";
    auto& memlet = *this->data_flow_graph_.out_edges(this->node_).begin();
    auto& out_container_type = types::infer_type(
        this->function_,
        this->function_.type(dynamic_cast<const data_flow::AccessNode&>(memlet.dst()).data()),
        memlet.subset());
    stream << this->language_extension_.zero(out_container_type.primitive_type()) << ";"
           << std::endl;

    // Create inner maps as for loops
    for (const std::pair<symbolic::Symbol, symbolic::Expression>& map : einsum_node->maps()) {
        if (std::find(einsum_node->out_indices().begin(), einsum_node->out_indices().end(),
                      map.first->get_name()) != einsum_node->out_indices().end())
            continue;
        stream << "for (" << map.first->get_name() << " = 0; " << map.first->get_name() << " < "
               << this->language_extension_.expression(map.second) << "; " << map.first->get_name()
               << "++)" << std::endl
               << "{" << std::endl;
        stream.setIndent(stream.indent() + 4);
        ++nested;
    }

    // Calculate one entry
    stream << this->containerSubset(einsum_node->output(0), einsum_node->out_indices()) << " = "
           << this->containerSubset(einsum_node->output(0), einsum_node->out_indices()) << " + ";
    for (size_t i = 0; i < einsum_node->inputs().size(); ++i) {
        if (i > 0) stream << " * ";
        stream << this->containerSubset(einsum_node->input(i), einsum_node->in_indices(i));
    }
    stream << ";" << std::endl;

    // Closing brackets
    for (size_t i = 0; i < nested; ++i) {
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(this->node_)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& type = this->function_.type(dst.data());
        stream << dst.data() << this->language_extension_.subset(function_, type, oedge.subset())
               << " = ";
        stream << oedge.src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace einsum
}  // namespace sdfg