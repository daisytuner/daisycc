#include "RegisterPasses.h"

#include <iostream>

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "polly/RegisterPasses.h"
#include "polly/CodePreparation.h"
#include "polly/Simplify.h"
#include "polly/ForwardOpTree.h"
#include "polly/DeLICM.h"
#include "polly/DeadCodeElimination.h"

#include "Scop2SDFG.h"

namespace daisy {

void registerDaisyPasses(llvm::PassBuilder &PB) {
    // Register daisy pass
    llvm::PassInstrumentationCallbacks *PIC = PB.getPassInstrumentationCallbacks();
    PB.registerPipelineParsingCallback(
        [PIC](llvm::StringRef Name, llvm::FunctionPassManager &FPM, llvm::ArrayRef<llvm::PassBuilder::PipelineElement> Pipeline) -> bool {
            if(Name != "Daisy"){
                return false;
            }

            bool UseMemSSA = true;
            FPM.addPass(llvm::PromotePass());
            FPM.addPass(llvm::EarlyCSEPass(UseMemSSA));
            FPM.addPass(llvm::InstCombinePass());
            FPM.addPass(llvm::SimplifyCFGPass());
            FPM.addPass(llvm::TailCallElimPass());
            FPM.addPass(llvm::SimplifyCFGPass());
            FPM.addPass(llvm::ReassociatePass());
            {
                llvm::LoopPassManager LPM;
                LPM.addPass(llvm::LoopRotatePass(true));
                FPM.addPass(llvm::createFunctionToLoopPassAdaptor<llvm::LoopPassManager>(
                    std::move(LPM), /*UseMemorySSA=*/false,
                    /*UseBlockFrequencyInfo=*/false));
            }
            FPM.addPass(llvm::InstCombinePass());
            {
                llvm::LoopPassManager LPM;
                LPM.addPass(llvm::IndVarSimplifyPass());
                FPM.addPass(llvm::createFunctionToLoopPassAdaptor<llvm::LoopPassManager>(
                    std::move(LPM), /*UseMemorySSA=*/false,
                    /*UseBlockFrequencyInfo=*/true));
            }

            // Common Polly Pipeline
            FPM.addPass(polly::CodePreparationPass());
            {
                polly::ScopPassManager SPM;
                SPM.addPass(polly::SimplifyPass(0));
                SPM.addPass(polly::ForwardOpTreePass());
                SPM.addPass(polly::DeLICMPass());
                SPM.addPass(polly::SimplifyPass(1));
                SPM.addPass(polly::DeadCodeElimPass());

                // ADDED: Scop2SDFG Pass
                SPM.addPass(daisy::Scop2SDFGPass());

                // Add SPM to pass
                FPM.addPass(polly::createFunctionToScopPassAdaptor(std::move(SPM)));
                FPM.addPass(llvm::SimplifyCFGPass());
            }
            return true;

        }
    );

}

}
