#pragma once

#include <benchmark/benchmark.h>

void RaiderSTREAM_Arguments(benchmark::internal::Benchmark* b);
void RaiderSTREAM_Arguments_Sequential(benchmark::internal::Benchmark* b);
void RaiderSTREAM_Arguments_Irregular(benchmark::internal::Benchmark* b);
