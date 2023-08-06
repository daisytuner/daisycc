#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include "RegisterPasses.h"

llvm::PassPluginLibraryInfo getDaisyPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "Daisy",
    LLVM_VERSION_STRING,
    daisy::registerDaisyPasses
  };
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getDaisyPluginInfo();
}
