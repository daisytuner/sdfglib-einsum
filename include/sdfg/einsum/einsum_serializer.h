#pragma once

#include <string>

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/serializer/library_node_serializer_registry.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace einsum {

/**
 * @brief Serializer for Einsum nodes
 *
 * This serializer is used to serialize and deserialize Einsum nodes.
 * It is registered with the LibraryNodeSerializerRegistry.
 */
class EinsumSerializer : public serializer::LibraryNodeSerializer {
   private:
    std::string expression(const symbolic::Expression& expr);

   public:
    virtual nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    virtual data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent) override;
};

// This function must be called by the application using the plugin
inline void register_einsum_serializer() {
    serializer::LibraryNodeSerializerRegistry::instance().register_library_node_serializer(
        LibraryNodeType_Einsum.value(), []() { return std::make_unique<EinsumSerializer>(); });
}

}  // namespace einsum
}  // namespace sdfg
