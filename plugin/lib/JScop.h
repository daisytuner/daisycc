#pragma once

#include "llvm/Support/JSON.h"

#include "polly/ScopInfo.h"
#include "polly/DependenceInfo.h"

namespace daisy {

    llvm::json::Value getJSON(polly::Scop &S, const polly::Dependences &D);

}

