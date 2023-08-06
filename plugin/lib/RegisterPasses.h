#pragma once

#include "llvm/Passes/PassBuilder.h"

namespace daisy {

    void registerDaisyPasses(llvm::PassBuilder &PB);

}
