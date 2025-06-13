#pragma once

#include <string>
#include <utility>
#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace einsum {

inline constexpr data_flow::LibraryNodeCode LibraryNodeType_Einsum("Einsum");

/**
 * @brief Einsum node
 *
 * This node enables the use of Einstein summation notation as library nodes in
 * the SDFG.
 */
class EinsumNode : public data_flow::LibraryNode {
   private:
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> maps_;

    std::vector<std::string> out_indices_;
    std::vector<std::vector<std::string>> in_indices_;

   public:
    EinsumNode(const DebugInfo& debug_info, const graph::Vertex vertex,
               data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
               const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
               const bool side_effect,
               const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps,
               const std::vector<std::string>& out_indices,
               const std::vector<std::vector<std::string>>& in_indices);

    EinsumNode(const EinsumNode&) = delete;
    EinsumNode& operator=(const EinsumNode&) = delete;

    virtual ~EinsumNode() = default;

    const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps() const;

    const std::pair<symbolic::Symbol, symbolic::Expression>& map(size_t index) const;

    const symbolic::Symbol& indvar(size_t index) const;

    const symbolic::Expression& num_iteration(size_t index) const;

    const std::vector<std::string>& out_indices() const;

    const std::string& out_index(size_t index) const;

    const std::vector<std::vector<std::string>>& in_indices() const;

    const std::vector<std::string>& in_indices(size_t index) const;

    const std::string& in_index(size_t index1, size_t index2) const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;

    virtual std::string toStr() const override;
};

}  // namespace einsum
}  // namespace sdfg
