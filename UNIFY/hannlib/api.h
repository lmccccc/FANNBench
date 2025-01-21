#pragma once

#include "core/base.h"
#include "core/hybrid_hnsw.h"
#include "extensions/attributes.h"
#include "extensions/scalar.h"
#include "extensions/spatial.h"

namespace hannlib
{

template <typename dist_t>
using ScalarHSIG = HSIG<dist_t, ScalarRangeExtension>;

}  // namespace hannlib