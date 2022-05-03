#!/usr/bin/env julia

# Copyright 2022 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
using Test, Pkg
using assignment16

@testset "Test assignment16" begin
    ans = fe_solver(100, x->x[1]^3 + 0.001)
    @test all(isapprox.(ans.free_values[1:13], [14.16908713140917, 13.341071592870716, 12.523240747303213, 11.726957540780601,
                          10.966032297651303, 10.254304105137516, 9.603083265327149, 9.019328675821981, 
                           8.50510092837698, 8.05820183075204, 7.673468306017411, 7.344148604459021, 7.0630084448622386], atol=1.0e-5))
end

