#include "sdfg/einsum/einsum_node.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace einsum {

EinsumNode::EinsumNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                       data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
                       const std::vector<std::string>& outputs,
                       const std::vector<std::string>& inputs, const bool side_effect,
                       const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps,
                       const std::vector<std::string>& out_indices,
                       const std::vector<std::vector<std::string>>& in_indices)
    : data_flow::LibraryNode(debug_info, vertex, parent, code, outputs, inputs, side_effect),
      maps_(maps),
      out_indices_(out_indices),
      in_indices_(in_indices) {
    // Check number of outputs
    if (outputs.size() != 1) {
        throw InvalidSDFGException("Einsum node can only have exactly one output");
    }

    // Check list sizes
    if (inputs.size() != in_indices.size()) {
        throw InvalidSDFGException("Number of input containers != number of input indices");
    }

    // Check if einsum indices are map indices
    bool missing = true;
    for (auto& indices : in_indices) {
        for (auto& index : indices) {
            missing = true;
            for (auto& map : maps) {
                if (index == map.first->get_name()) {
                    missing = false;
                    break;
                }
            }
            if (missing) {
                throw InvalidSDFGException("Einsum index " + index + " not found in map indices");
            }
        }
    }
    for (auto& index : out_indices) {
        missing = true;
        for (auto& map : maps) {
            if (index == map.first->get_name()) {
                missing = false;
                break;
            }
        }
        if (missing) {
            throw InvalidSDFGException("Einsum index " + index + " not found in map indices");
        }
    }

    // TODO: Check if container exist and types match einsum index access
    // The Problem: For a types::infer_type, I need a sdfg::Function which I am unable to get at
    //              this point
}

const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& EinsumNode::maps() const {
    return this->maps_;
}

const std::pair<symbolic::Symbol, symbolic::Expression>& EinsumNode::map(size_t index) const {
    return this->maps_[index];
}

const symbolic::Symbol& EinsumNode::indvar(size_t index) const { return this->maps_[index].first; }

const symbolic::Expression& EinsumNode::num_iteration(size_t index) const {
    return this->maps_[index].second;
}

const std::vector<std::string>& EinsumNode::out_indices() const { return this->out_indices_; }

const std::string& EinsumNode::out_index(size_t index) const { return this->out_indices_[index]; }

const std::vector<std::vector<std::string>>& EinsumNode::in_indices() const {
    return this->in_indices_;
}

const std::vector<std::string>& EinsumNode::in_indices(size_t index) const {
    return this->in_indices_[index];
}

const std::string& EinsumNode::in_index(size_t index1, size_t index2) const {
    return this->in_indices_[index1][index2];
}

std::unique_ptr<data_flow::DataFlowNode> EinsumNode::clone(const graph::Vertex vertex,
                                                           data_flow::DataFlowGraph& parent) const {
    return std::make_unique<EinsumNode>(this->debug_info(), vertex, parent, this->code(),
                                        this->outputs(), this->inputs(), this->side_effect(),
                                        this->maps(), this->out_indices(), this->in_indices());
}

void EinsumNode::replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) {
    // Do nothing
}

std::string EinsumNode::toStr() const {
    std::stringstream stream;

    stream << this->outputs_[0];
    if (this->out_indices_.size() > 0) {
        stream << "[";
        for (size_t i = 0; i < this->out_indices_.size(); ++i) {
            if (i > 0) stream << ",";
            stream << this->out_indices_[i];
        }
        stream << "]";
    }
    stream << " = ";
    for (size_t i = 0; i < this->inputs_.size(); ++i) {
        if (i > 0) stream << " * ";
        stream << this->inputs_[i];
        if (this->in_indices_[i].size() > 0) {
            stream << "[";
            for (size_t j = 0; j < this->in_indices_[i].size(); j++) {
                if (j > 0) stream << ",";
                stream << this->in_indices_[i][j];
            }
            stream << "]";
        }
    }

    for (auto& map : this->maps_) {
        stream << " for " << map.first->__str__() << " = 0:" << map.second->__str__();
    }

    return stream.str();
}

}  // namespace einsum
}  // namespace sdfg