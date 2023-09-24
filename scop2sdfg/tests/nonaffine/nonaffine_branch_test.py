from scop2sdfg.ir import Scop
from scop2sdfg.codegen import Generator


def test_nonaffine_branch():
    jscop = {
        "access_range": [],
        "arrays": [
            {
                "kind": "array",
                "name": "MemRef0",
                "sizes": ["*"],
                "type": "double",
                "variable": "  %A = alloca [256 x double], align 16",
            },
            {
                "kind": "array",
                "name": "MemRef1",
                "sizes": ["*"],
                "type": "double",
                "variable": "  %B = alloca [256 x double], align 16",
            },
            {
                "kind": "array",
                "name": "MemRef3",
                "sizes": ["*"],
                "type": "double",
                "variable": "  %C = alloca [256 x double], align 16",
            },
            {
                "kind": "phi",
                "name": "MemRef2__phi",
                "sizes": [],
                "type": "double",
                "variable": "  %1 = phi double [ %.pre, %for.body11.if.end_crit_edge ], [ %add, %if.then ]",
            },
        ],
        "context": "{  :  }",
        "dependencies": {
            "RAW": "{ Stmt0[i0] -> Stmt1[i0] : 0 <= i0 <= 255 }",
            "RED": "{  }",
            "TC_RED": "{  }",
            "WAR": "{  }",
            "WAW": "{ Stmt0[i0] -> Stmt1[i0] : 0 <= i0 <= 255 }",
        },
        "instructions": "  %indvars.iv55 = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next56, %if.end ]\\n  %arrayidx13 = getelementptr inbounds [256 x double], ptr %A, i64 0, i64 %indvars.iv55\\n  %0 = load double, ptr %arrayidx13, align 8, !tbaa !5\\n  %cmp14 = fcmp ogt double %0, 0.000000e+00\\n  br i1 %cmp14, label %if.then, label %for.body11.if.end_crit_edge\\n  %add = fadd double %0, 2.000000e+00\\n  %arrayidx19 = getelementptr inbounds [256 x double], ptr %B, i64 0, i64 %indvars.iv55\\n  store double %add, ptr %arrayidx19, align 8, !tbaa !5\\n  br label %if.end\\n  %1 = phi double [ %.pre, %for.body11.if.end_crit_edge ], [ %add, %if.then ]\\n  %add22 = fadd double %1, 1.000000e+00\\n  %arrayidx24 = getelementptr inbounds [256 x double], ptr %C, i64 0, i64 %indvars.iv55\\n  store double %add22, ptr %arrayidx24, align 8, !tbaa !5\\n  %indvars.iv.next56 = add nuw nsw i64 %indvars.iv55, 1\\n  %exitcond58.not = icmp eq i64 %indvars.iv.next56, 256\\n  br i1 %exitcond58.not, label %for.body33.preheader, label %for.body11, !llvm.loop !12\\n  br label %for.body33\\n  %arrayidx21.phi.trans.insert = getelementptr inbounds [256 x double], ptr %B, i64 0, i64 %indvars.iv55\\n  %.pre = load double, ptr %arrayidx21.phi.trans.insert, align 8, !tbaa !5\\n  br label %if.end\\n",
        "name": "%for.body11---%for.body33",
        "parameters": [],
        "schedule": "{ Stmt1[i0] -> [i0, 1]; Stmt0[i0] -> [i0, 0] }",
        "statements": [
            {
                "accesses": [
                    {
                        "access_instruction": "  %0 = load double, ptr %arrayidx13, align 8, !tbaa !5",
                        "incoming_value": "",
                        "kind": "read",
                        "relation": "{ Stmt0[i0] -> MemRef0[i0] }",
                    },
                    {
                        "access_instruction": "  store double %add, ptr %arrayidx19, align 8, !tbaa !5",
                        "incoming_value": "  %add = fadd double %0, 2.000000e+00",
                        "kind": "write",
                        "relation": "{ Stmt0[i0] -> MemRef1[i0] }",
                    },
                    {
                        "access_instruction": "  %.pre = load double, ptr %arrayidx21.phi.trans.insert, align 8, !tbaa !5",
                        "incoming_value": "",
                        "kind": "read",
                        "relation": "{ Stmt0[i0] -> MemRef1[i0] }",
                    },
                    {
                        "access_instruction": "  %1 = phi double [ %.pre, %for.body11.if.end_crit_edge ], [ %add, %if.then ]",
                        "incoming_value": "",
                        "kind": "write",
                        "relation": "{ Stmt0[i0] -> MemRef3[i0] }",
                    },
                ],
                "affine": "false",
                "domain": "{ Stmt0[i0] : 0 <= i0 <= 255 }",
                "loops": [
                    {
                        "induction_variable": "  %indvars.iv55 = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next56, %if.end ]"
                    }
                ],
                "name": "Stmt0",
            },
            {
                "accesses": [
                    {
                        "access_instruction": "  %1 = phi double [ %.pre, %for.body11.if.end_crit_edge ], [ %add, %if.then ]",
                        "incoming_value": "",
                        "kind": "read",
                        "relation": "{ Stmt1[i0] -> MemRef3[i0] }",
                    },
                    {
                        "access_instruction": "  store double %add22, ptr %arrayidx24, align 8, !tbaa !5",
                        "incoming_value": "  %add22 = fadd double %1, 1.000000e+00",
                        "kind": "write",
                        "relation": "{ Stmt1[i0] -> MemRef3[i0] }",
                    },
                ],
                "affine": "true",
                "domain": "{ Stmt1[i0] : 0 <= i0 <= 255 }",
                "loops": [
                    {
                        "induction_variable": "  %indvars.iv55 = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next56, %if.end ]"
                    }
                ],
                "name": "Stmt1",
            },
        ],
    }
    scop = Scop.from_json(source="", desc=jscop)
    scop.validate()

    found = False
    for _, acc in scop._memory_accesses["Stmt0"].items():
        if (
            acc.instruction
            == "%1 = phi double [ %.pre, %for.body11.if.end_crit_edge ], [ %add, %if.then ]"
        ):
            assert acc.kind == "write"
            assert len(acc.arguments()) == 1

            arg = next(iter(acc.arguments()))
            assert (
                scop._computations[arg.reference].code
                == "select i1 %cmp14, double %.pre, double %add"
            )

            found = True
            break

    assert found

    sdfg = Generator.generate(scop)
    sdfg.validate()
