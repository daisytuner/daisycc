#pragma once

#include <iostream>
#include <filesystem>
#include <stdlib.h>
#include <algorithm>


#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include "polly/ScopPass.h"

#include "JScop.h"

static bool DaisyTransferTune;
static llvm::cl::opt<bool, true> XTransferTune(
    "daisy-transfer-tune",
    llvm::cl::location(DaisyTransferTune),
    llvm::cl::desc("Process"),
    llvm::cl::init(true)
);

namespace daisy {

namespace fs = std::filesystem;

struct Scop2SDFGPass : public llvm::PassInfoMixin<Scop2SDFGPass> {
private:

    static bool scop2sdfg(const std::string source_path, const std::string jscop_str) {
        std::string sdfg_name;
        if (!system(NULL))
            return false;

        std::string command = "scop2sdfg --source_path=";
        command += source_path;
        command += " --scop='";
        command += jscop_str;
        command += "'";
        if (DaisyTransferTune) {
            command += " --transfer_tune";
        }

        return (system(command.c_str()) == 0);
    }

    static bool hasEscapingValue(polly::Scop& S) {
        for (auto* bb : S.getRegion().blocks()) {
            for (llvm::BasicBlock::iterator I = bb->begin(); I != bb->end(); ++I) {
                if (S.isEscaping(&(*I))) {
                    return true;
                }
            }
        }
        return false;
    }

    static bool hasNonIntegralParameters(polly::Scop& S) {
        for (auto& param : S.parameters()) {
            if (!param->getType()->isIntegerTy()) {
                return false;
            }
        }
        return true;
    }

    static bool can_be_applied(polly::Scop& S) {
        if (hasEscapingValue(S)) {
            llvm::errs() << "has escaping value\n";
            return false;
        }

        if (!hasNonIntegralParameters(S)) {
            llvm::errs() << "has non-integral parameter value\n";
            return false;
        }

        return true;
    }

public:
    llvm::PreservedAnalyses run(polly::Scop& S, polly::ScopAnalysisManager &SAM, polly::ScopStandardAnalysisResults &SAR, polly::SPMUpdater &U)
    {
        // Gather contextual information
        llvm::Region& region = S.getRegion();
        llvm::Function& function = S.getFunction();
        llvm::LLVMContext& context = function.getContext();
        llvm::Module* current_module = function.getParent();
        fs::path source_path{current_module->getSourceFileName()};

        llvm::errs() << "Scop: " << S.getNameStr() << " " << function.getName() << "\n";
        if (!can_be_applied(S)) {
            llvm::errs() << "Cannot be applied\n";
            return llvm::PreservedAnalyses::all();
        }
        
        // Scop to JSON
        const polly::Dependences &D = SAM.getResult<polly::DependenceAnalysis>(S, SAR).getDependences(polly::Dependences::AL_Statement);
        std::string jscop_str;
        try {
            llvm::json::Value jscop = daisy::getJSON(S, D);
            llvm::raw_string_ostream output(jscop_str);
            output << jscop;
        } catch (std::invalid_argument const& ex) {
            std::cout << ex.what() << std::endl;
            return llvm::PreservedAnalyses::all();
        }

        // Call scop2sdfg python module
        bool success = scop2sdfg(source_path, jscop_str);
        if (!success) {
            return llvm::PreservedAnalyses::all();
        }
        llvm::errs() << "Scop2SDFG successful\n";

        // Parse SDFG name. Has to by in sync with python module (move to CLI)
        std::string sdfg_name = "sdfg_" + source_path.filename().u8string() + "_" + S.getNameStr();
        sdfg_name.erase(std::remove(sdfg_name.begin(), sdfg_name.end(), '.'), sdfg_name.end());
        sdfg_name.erase(std::remove(sdfg_name.begin(), sdfg_name.end(), '%'), sdfg_name.end());
        std::replace(sdfg_name.begin(), sdfg_name.end(), '-', '_');

        // Declare SDFG functions
        llvm::StructType* sdfg_type = llvm::StructType::create(context, sdfg_name);
        llvm::PointerType* sdfg_type_ptr = llvm::PointerType::getUnqual(sdfg_type);

        // Init SDFG
        std::vector<llvm::Type*> init_args;
        for (auto& param : S.parameters()) {
            init_args.push_back(param->getType());
        }
        llvm::FunctionType *init_sdfg_func_type = llvm::FunctionType::get(sdfg_type_ptr, init_args, false);
        llvm::Function *init_sdfg_func_decl = llvm::Function::Create(init_sdfg_func_type, llvm::Function::ExternalLinkage, "__dace_init_" + sdfg_name, current_module);

        // Exit SDFG
        std::vector<llvm::Type*> exit_args = {
            sdfg_type_ptr
        };
        llvm::FunctionType *exit_sdfg_func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), exit_args, false);
        llvm::Function *exit_sdfg_func_decl = llvm::Function::Create(exit_sdfg_func_type, llvm::Function::ExternalLinkage, "__dace_exit_" + sdfg_name, current_module);

        std::vector<polly::ScopArrayInfo*> arrays;
        for (auto SAI : S.arrays()) {
            if (SAI->getKind() != polly::MemoryKind::Array)
                continue;

            arrays.push_back(SAI);
        }
        sort(arrays.begin(), arrays.end(), [&](polly::ScopArrayInfo* t1, polly::ScopArrayInfo* t2) {
            return t1->getName() < t2->getName();
        });

        std::vector<polly::ScopArrayInfo*> scalars;
        for (auto SAI : S.arrays()) {
            if (SAI->getKind() != polly::MemoryKind::Value)
                continue;

            scalars.push_back(SAI);
        }
        sort(scalars.begin(), scalars.end(), [&](polly::ScopArrayInfo* t1, polly::ScopArrayInfo* t2) {
            return t1->getName() < t2->getName();
        });

        // Program
        std::vector<llvm::Type*> program_args = {
            // State
            sdfg_type_ptr,
        };
        for (auto SAI : arrays) {
            program_args.push_back(SAI->getBasePtr()->getType());
        }
        for (auto SAI : scalars) {
            program_args.push_back(SAI->getBasePtr()->getType());
        }
        for (auto& param : S.parameters()) {
            program_args.push_back(param->getType());
        }
        llvm::FunctionType *program_sdfg_func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), program_args, false);
        llvm::Function *program_sdfg_func_decl = llvm::Function::Create(program_sdfg_func_type, llvm::Function::ExternalLinkage, "__program_" + sdfg_name, current_module);

        // // Re-direct entering and exiting blocks
        llvm::BasicBlock* entering_block = S.getEnteringBlock();
        llvm::BasicBlock* exiting_block = S.getExitingBlock();
        llvm::BasicBlock* exit_block = S.getExit();

        // Create new IRBuilder for adding calls to SDFG
        llvm::BasicBlock* daceblock = llvm::BasicBlock::Create(context, "daceblock", &function); 
        llvm::IRBuilder<> builder(daceblock);
        builder.SetInsertPoint(daceblock);

        std::vector<llvm::Value*> init_vals;
        for (auto& param : S.parameters()) {
            llvm::Value* value = nullptr;
            const llvm::SCEVUnknown* unknown = llvm::dyn_cast_or_null<llvm::SCEVUnknown>(param);
            if (unknown) {
                value = unknown->getValue();
            } else {
                const llvm::SCEVAddRecExpr* rec = llvm::dyn_cast_or_null<llvm::SCEVAddRecExpr>(param);
                value = rec->getLoop()->getInductionVariable(*S.getSE());
            }
            init_vals.push_back(value);
        }
        llvm::CallInst* init_call = builder.CreateCall(init_sdfg_func_decl, init_vals, sdfg_name + "_state");

        std::vector<llvm::Value*> program_vals = {
            init_call
        };
        for (auto SAI : arrays) {
            program_vals.push_back(SAI->getBasePtr());
        }
        for (auto SAI : scalars) {
            program_vals.push_back(SAI->getBasePtr());
        }
        for (auto& param : S.parameters()) {
            llvm::Value* value = nullptr;
            const llvm::SCEVUnknown* unknown = llvm::dyn_cast_or_null<llvm::SCEVUnknown>(param);
            if (unknown) {
                value = unknown->getValue();
            } else {
                const llvm::SCEVAddRecExpr* rec = llvm::dyn_cast_or_null<llvm::SCEVAddRecExpr>(param);
                value = rec->getLoop()->getInductionVariable(*S.getSE());
            }
            program_vals.push_back(value);
        }
        llvm::CallInst* program_call = builder.CreateCall(program_sdfg_func_decl, program_vals);

        std::vector<llvm::Value*> exit_vals = {
            init_call
        };
        llvm::CallInst* exit_call = builder.CreateCall(exit_sdfg_func_decl, exit_vals);

        // Connect entry to daceblock
        entering_block->getTerminator()->setSuccessor(0, daceblock);

        // Connect daceblock to exit
        llvm::BranchInst *end = builder.CreateBr(exit_block);
        for (auto& phi : exit_block->phis()) {
            for (int i = 0; i < phi.getNumIncomingValues(); i++) {
                if (phi.getIncomingBlock(i) == exiting_block) {
                    phi.addIncoming(phi.getIncomingValue(i), daceblock);
                    break;
                }
            }
        }

        S.markAsToBeSkipped();
        return llvm::PreservedAnalyses::none();
    }

    static bool isRequired()
    {
        return true;
    }
};

}