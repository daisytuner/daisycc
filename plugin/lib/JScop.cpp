#include "JScop.h"

#include <iostream>
#include <sstream>

#include "isl/set.h"

#include "polly/Support/ScopLocation.h"
#include "polly/Support/GICHelper.h"

#include "llvm/Analysis/MemorySSA.h"

namespace daisy {
    /**
     * LLVM IR is structured as follows:
     * - Instructions are SSA.
     * - Basic blocks are sequences of instructions, which execute sequentially (no branches).
     * - The control-flow graph (CFG) visualizes the jumps between basic blocks.
     * - Regions are subgraphs of the CFG such that there is only a single entry edge and a single exit edge in the CFG.
     * - A region may contain a sub-region.
     * - Regions may have sibling regions (share same entry exit, but different blocks).
     * 
     * Scops are structured as follows:
     * - A Scop is a maximal region in polyhedral representation
     * - A Scop consists of integer parameters and their domain, and statements.
     * - A statement represents a single basic block (simple; non-affine case).
    */

    llvm::json::Array defineParameters(polly::Scop &S) {
        llvm::json::Array parameters;
        for (auto& param : S.parameters()) {
            llvm::json::Object parameter;

            parameter["name"] = polly::stringFromIslObj(S.getIdForParam(param));

            llvm::Value* value = nullptr;
            const llvm::SCEVUnknown* unknown = llvm::dyn_cast_or_null<llvm::SCEVUnknown>(param);
            if (unknown) {
                value = unknown->getValue();
            } else {
                const llvm::SCEVAddRecExpr* rec = llvm::dyn_cast_or_null<llvm::SCEVAddRecExpr>(param);
                if (!rec) {
                    throw std::invalid_argument("Failed to parse parameter");
                }
                value = rec->getLoop()->getInductionVariable(*S.getSE());
            }

            std::string param_str;
            llvm::raw_string_ostream param_stream(param_str);
            param_stream << *value;
            parameter["variable"] = param_str;

            std::string type_str;
            llvm::raw_string_ostream type_os(type_str);
            type_os << *param->getType();
            parameter["type"] = type_str;

            parameters.push_back(std::move(parameter));
        }
        return parameters;
    }

    llvm::json::Array defineArrays(polly::Scop &S) {
        llvm::json::Array Arrays;
        std::string Buffer;
        llvm::raw_string_ostream RawStringOstream(Buffer);

        std::vector<polly::ScopArrayInfo*> arrays;
        for (auto SAI : S.arrays()) {
            if (SAI->getKind() != polly::MemoryKind::Array)
                continue;
            arrays.push_back(SAI);
        }
        sort(arrays.begin(), arrays.end(), [&](polly::ScopArrayInfo* t1, polly::ScopArrayInfo* t2) {
            return t1->getName() < t2->getName();
        });
        for (auto &SAI : arrays) {
            llvm::json::Object Array;

            Array["name"] = SAI->getName();

            if (SAI->getKind() == polly::MemoryKind::Array) {
                Array["kind"] = "array";
            } else if (SAI->getKind() == polly::MemoryKind::Value) {
                Array["kind"] = "value";
            } else if (SAI->getKind() == polly::MemoryKind::PHI) {
                Array["kind"] = "phi";
            } else {
                Array["kind"] = "exit_phi";
            }

            SAI->getElementType()->print(RawStringOstream);
            Array["type"] = RawStringOstream.str();
            Buffer.clear();
            
            llvm::json::Array Sizes;
            for (int i = 0; i < SAI->getNumberOfDimensions(); i++) {
                auto size = SAI->getDimensionSize(i);
                if (!size) {
                    Sizes.push_back("*");
                } else {
                    size->print(RawStringOstream);
                    Sizes.push_back(RawStringOstream.str());
                    Buffer.clear();
                }
            }
            Array["sizes"] = std::move(Sizes);

            SAI->getBasePtr()->print(RawStringOstream);
            Array["variable"] = RawStringOstream.str();
            Buffer.clear();

            Arrays.push_back(std::move(Array));
        }

        std::vector<polly::ScopArrayInfo*> scalars;
        for (auto SAI : S.arrays()) {
            if (SAI->getKind() != polly::MemoryKind::Value)
                continue;
            scalars.push_back(SAI);
        }
        sort(scalars.begin(), scalars.end(), [&](polly::ScopArrayInfo* t1, polly::ScopArrayInfo* t2) {
            return t1->getName() < t2->getName();
        });
        for (auto &SAI : scalars) {
            llvm::json::Object Array;

            Array["name"] = SAI->getName();

            if (SAI->getKind() == polly::MemoryKind::Array) {
                Array["kind"] = "array";
            } else if (SAI->getKind() == polly::MemoryKind::Value) {
                Array["kind"] = "value";
            } else if (SAI->getKind() == polly::MemoryKind::PHI) {
                Array["kind"] = "phi";
            } else {
                Array["kind"] = "exit_phi";
            }

            SAI->getElementType()->print(RawStringOstream);
            Array["type"] = RawStringOstream.str();
            Buffer.clear();
            
            llvm::json::Array Sizes;
            for (int i = 0; i < SAI->getNumberOfDimensions(); i++) {
                auto size = SAI->getDimensionSize(i);
                if (!size) {
                    Sizes.push_back("*");
                } else {
                    size->print(RawStringOstream);
                    Sizes.push_back(RawStringOstream.str());
                    Buffer.clear();
                }
            }
            Array["sizes"] = std::move(Sizes);

            SAI->getBasePtr()->print(RawStringOstream);
            Array["variable"] = RawStringOstream.str();
            Buffer.clear();

            Arrays.push_back(std::move(Array));
        }

        std::vector<polly::ScopArrayInfo*> phis;
        for (auto SAI : S.arrays()) {
            if (SAI->getKind() == polly::MemoryKind::Value || SAI->getKind() == polly::MemoryKind::Array)
                continue;
            phis.push_back(SAI);
        }
        sort(phis.begin(), phis.end(), [&](polly::ScopArrayInfo* t1, polly::ScopArrayInfo* t2) {
            return t1->getName() < t2->getName();
        });
        for (auto &SAI : phis) {
            llvm::json::Object Array;

            Array["name"] = SAI->getName();

            if (SAI->getKind() == polly::MemoryKind::Array) {
                Array["kind"] = "array";
            } else if (SAI->getKind() == polly::MemoryKind::Value) {
                Array["kind"] = "value";
            } else if (SAI->getKind() == polly::MemoryKind::PHI) {
                Array["kind"] = "phi";
            } else {
                Array["kind"] = "exit_phi";
            }

            SAI->getElementType()->print(RawStringOstream);
            Array["type"] = RawStringOstream.str();
            Buffer.clear();
            
            llvm::json::Array Sizes;
            for (int i = 0; i < SAI->getNumberOfDimensions(); i++) {
                auto size = SAI->getDimensionSize(i);
                if (!size) {
                    Sizes.push_back("*");
                } else {
                    size->print(RawStringOstream);
                    Sizes.push_back(RawStringOstream.str());
                    Buffer.clear();
                }
            }
            Array["sizes"] = std::move(Sizes);

            SAI->getBasePtr()->print(RawStringOstream);
            Array["variable"] = RawStringOstream.str();
            Buffer.clear();

            Arrays.push_back(std::move(Array));
        }

        return Arrays;
    }

    llvm::json::Value getJSON(polly::Scop &S, const polly::Dependences &D) {
        llvm::json::Object root;

        // Define the function
        root["name"] = S.getNameStr();
        root["parameters"] = defineParameters(S);
        root["arrays"] = defineArrays(S);

        std::string insts_str;
        llvm::raw_string_ostream insts_os(insts_str);
        for (auto* bb : S.getRegion().blocks()) {
            for (llvm::BasicBlock::iterator I = bb->begin(); I != bb->end(); ++I) {
                insts_os << *I << "\n";
            }
        }
        root["instructions"] = insts_str;

        // Define the ISL AST

        root["context"] = S.getContextStr();
        root["schedule"] = polly::stringFromIslObj(S.getSchedule());

        root["dependencies"];
        llvm::json::Object dependencies;
        
        auto raw = D.getDependences(polly::Dependences::Type::TYPE_RAW);
        std::stringstream ss_raw;
        ss_raw << raw;
        dependencies["RAW"] = ss_raw.str();

        auto war = D.getDependences(polly::Dependences::Type::TYPE_WAR);
        std::stringstream ss_war;
        ss_war << war;
        dependencies["WAR"] = ss_war.str();

        auto waw = D.getDependences(polly::Dependences::Type::TYPE_WAW);
        std::stringstream ss_waw;
        ss_waw << waw;
        dependencies["WAW"] = ss_waw.str();

        auto red = D.getDependences(polly::Dependences::Type::TYPE_RED);
        std::stringstream ss_red;
        ss_red << red;
        dependencies["RED"] = ss_red.str();

        auto tc_red = D.getDependences(polly::Dependences::Type::TYPE_TC_RED);
        std::stringstream ss_tc_red;
        ss_tc_red << tc_red;
        dependencies["TC_RED"] = ss_tc_red.str();

        root["dependencies"] = std::move(dependencies);

        /**
         * Statement:
         *  - name: name of statement
         *  - domain: parameter space of statement
         *  - basic_block: instructions representing the statement
         *  - accesses: memory accesses of statement
        */
        llvm::json::Array Statements;
        for (polly::ScopStmt &Stmt : S) {
            llvm::json::Object statement;

            statement["name"] = Stmt.getBaseName();
            statement["domain"] = Stmt.getDomainStr();
            statement["affine"] = Stmt.isBlockStmt();

            // Loop info
            llvm::json::Array loops;
            for (size_t i = 0; i < Stmt.getNumIterators(); i++) {
                llvm::Loop* surrounding_loop = Stmt.getLoopForDimension(i);
                if (surrounding_loop == nullptr) {
                    continue;
                }

                llvm::json::Object loop;

                // Induction variable
                std::string indvar_str;
                llvm::PHINode* indvar = surrounding_loop->getInductionVariable(*S.getSE());
                if (indvar != nullptr) {
                    llvm::raw_string_ostream indvar_os(indvar_str);
                    indvar_os << *indvar;
                }
                loop["induction_variable"] = indvar_str;
                loops.push_back(std::move(loop));
            }
            statement["loops"] = std::move(loops);

            // Accesses
            llvm::json::Array Accesses;
            for (polly::MemoryAccess *MA : Stmt) {
                llvm::json::Object access;

                access["kind"] = MA->isRead() ? "read" : "write";
                access["relation"] = MA->getAccessRelationStr();
            
                std::string access_inst;
                llvm::raw_string_ostream os_access_inst(access_inst);
                if (MA->getAccessInstruction() != nullptr) {
                    os_access_inst << *MA->getAccessInstruction();
                }
                access["access_instruction"] = access_inst;

                std::string incoming_value;
                llvm::raw_string_ostream os_incoming_value(incoming_value);
                if (MA->isWrite() && MA->tryGetValueStored() != nullptr) {
                    os_incoming_value << *MA->tryGetValueStored();
                }
                access["incoming_value"] = incoming_value;

                Accesses.push_back(std::move(access));
            }
            statement["accesses"] = std::move(Accesses);

            Statements.push_back(std::move(statement));
        }
        root["statements"] = std::move(Statements);

        // Add shape analysis
        llvm::json::Array alias_groups;
        for (auto& alias_group : S.getAliasGroups()) {
            llvm::json::Object group;

            llvm::json::Array members_rw;
            for (auto& rw : alias_group.first) {
                llvm::json::Object member;
                member["minimal"] = polly::stringFromIslObj(rw.first);
                member["maximal"] = polly::stringFromIslObj(rw.second);
                members_rw.push_back(std::move(member));
            }
            group["readwrite"] = std::move(members_rw);
            
            llvm::json::Array members_ro;            
            for (auto& ro : alias_group.second) {
                llvm::json::Object member;
                member["minimal"] = polly::stringFromIslObj(ro.first);
                member["maximal"] = polly::stringFromIslObj(ro.second);
                members_ro.push_back(std::move(member));
            }
            group["readonly"] = std::move(members_ro);
            alias_groups.push_back(std::move(group));
        }
        root["access_range"] = std::move(alias_groups);

        return llvm::json::Value(std::move(root));
    }

}

