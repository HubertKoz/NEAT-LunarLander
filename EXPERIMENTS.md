# Fitness Strategy Experiments / Eksperymenty strategii fitnessu

This document summarizes the experiments performed for the NEAT LunarLander project.  
Dokument podsumowuje eksperymenty wykonane w projekcie NEAT LunarLander.

---

## 1. Experimental setup / Konfiguracja eksperymentów

All experiments were run in the `LunarLander-v3` environment using the NEAT neuroevolution algorithm.

Wszystkie eksperymenty zostały przeprowadzone w środowisku `LunarLander-v3` z użyciem algorytmu neuroewolucyjnego NEAT.

The common configuration was:

| Parameter | Value |
|---|---:|
| Maximum steps per episode | 500 |
| Evaluation runs per genome | 3 |
| Fixed seeds | `[42, 123, 999]` |
| Population size | 150 |
| Maximum generations | 300 |
| Network type | Feed-forward |
| Early fitness termination | Disabled |

In each generation, every genome was evaluated on three episodes. The final fitness assigned to a genome was the average fitness over these runs. Fixed seeds were used to make the comparison between strategies more stable and reproducible.

W każdej generacji każdy genom był oceniany w trzech epizodach. Końcowy fitness przypisany genomowi był średnią z tych trzech prób. Użyto stałych seedów, aby porównanie między strategiami było bardziej stabilne i powtarzalne.

The option `no_fitness_termination = True` was used in `config/neat_config.txt`, so all strategies were trained for the full 300 generations. This makes the training curves easier to compare.

W pliku `config/neat_config.txt` ustawiono `no_fitness_termination = True`, dlatego każda strategia była trenowana przez pełne 300 generacji. Dzięki temu przebiegi treningu są łatwiejsze do porównania.

---

## 2. Tested fitness strategies / Testowane strategie fitnessu

| Strategy | Description |
|---|---|
| `default` | Native Gymnasium reward without modification. |
| `penalty_time` | Native reward with an additional constant time penalty at each step. |
| `penalty_angle` | Native reward with an additional penalty for tilted ground contact. |
| `landing_quality` | Reward shaping based on distance from the center, velocity, angle and stable two-leg contact. |
| `centered_landing` | Reward shaping focused mostly on staying close to the center of the landing area. |

---

## 3. Results / Wyniki

The following table was generated from `results/summary.csv`.

Poniższa tabela została przygotowana na podstawie pliku `results/summary.csv`.

| Strategy | Generations | Best fitness | Best generation | Final mean fitness | First generation >= 200 |
|---|---:|---:|---:|---:|---:|
| `default` | 300 | 241.59 | 290 | -105.76 | 171 |
| `penalty_time` | 300 | 192.10 | 262 | -113.57 | — |
| `penalty_angle` | 300 | 45.07 | 298 | -227.89 | — |
| `landing_quality` | 300 | 2061.85 | 297 | -896.17 | 6 |
| `centered_landing` | 300 | -383.99 | 244 | -831.13 | — |

The comparison plot is stored in:

```text
results/best_fitness_comparison.png

```

---

# Interpretation of the results / Interpetacja wyników

**ENG version:**

The default strategy achieved the best result in the native Gymnasium reward scale. Its best fitness was 241.59, reached near the end of training, and it crossed the conventional 200-point threshold for the first time at generation 171. This makes it the strongest result if we judge the agent using the original LunarLander reward.

The penalty_time strategy reached a best fitness of 192.10. It performed much better than the angle-based and centered strategies, but it did not cross the 200-point threshold. This suggests that a small constant time penalty can still produce a reasonably good policy, but in this run it did not improve over the default reward.

The penalty_angle strategy reached only 45.07 best fitness. Although the goal of this strategy was to encourage more vertical and stable landings, the additional angle penalty seems to have made optimization harder. The final mean fitness also remained clearly negative, which means most genomes in the population still performed poorly.

The landing_quality strategy achieved the highest numerical fitness, with a best value of 2061.85. However, this score should not be compared directly with the native Gymnasium reward, because the strategy changes the fitness scale by adding extra shaping terms and a bonus for two-leg contact. Therefore, this result shows that NEAT learned to exploit or optimize this shaped objective, but it does not automatically mean that the original LunarLander task was solved better than with the default strategy.

The centered_landing strategy performed the worst, with a best fitness of -383.99. The strong penalty for horizontal distance from the center probably made the objective too restrictive. Instead of helping the agent, it may have penalized many trajectories before the agent learned basic landing control.

Overall, the default reward produced the best result in the original environment scale. The additional shaped rewards changed the behavior of the optimization process, but they did not clearly improve performance under the original LunarLander metric. The experiment also shows that reward shaping must be designed carefully, because additional penalties or bonuses can change the fitness scale and make raw scores difficult to compare.


**PL version:**

Strategia default osiągnęła najlepszy wynik w oryginalnej skali nagrody środowiska Gymnasium. Najlepszy fitness wyniósł 241.59 i został osiągnięty pod koniec treningu. Strategia po raz pierwszy przekroczyła umowny próg 200 punktów w generacji 171. Jeśli oceniamy agenta według oryginalnej metryki LunarLandera, to właśnie default dał najlepszy wynik.

Strategia penalty_time osiągnęła najlepszy fitness równy 192.10. Wynik był wyraźnie lepszy niż dla penalty_angle i centered_landing, ale nie przekroczył progu 200 punktów. Sugeruje to, że niewielka stała kara za czas może nadal prowadzić do sensownej polityki, ale w tym eksperymencie nie poprawiła wyniku względem domyślnej funkcji nagrody.

Strategia penalty_angle osiągnęła jedynie 45.07 najlepszego fitnessu. Chociaż celem tej strategii było promowanie bardziej pionowego i stabilnego lądowania, dodatkowa kara za przechylenie najwyraźniej utrudniła optymalizację. Średni fitness populacji pozostał ujemny, więc większość genomów nadal radziła sobie słabo.

Strategia landing_quality osiągnęła najwyższy wynik liczbowy, czyli 2061.85. Tego wyniku nie należy jednak porównywać bezpośrednio z wynikiem strategii default, ponieważ landing_quality zmienia skalę fitnessu przez dodanie własnych kar i bonusu za kontakt obu nóg z podłożem. Ten eksperyment pokazuje więc, że NEAT nauczył się optymalizować zmodyfikowaną funkcję celu, ale nie oznacza automatycznie, że oryginalne środowisko LunarLander zostało rozwiązane lepiej niż przy strategii domyślnej.

Strategia centered_landing wypadła najsłabiej, osiągając najlepszy fitness równy -383.99. Silna kara za odległość od środka lądowiska prawdopodobnie sprawiła, że funkcja celu była zbyt restrykcyjna. Zamiast pomóc agentowi, mogła karać wiele trajektorii zanim agent nauczył się podstawowej kontroli lądownika.

Ogólnie najlepszym wynikiem w oryginalnej skali środowiska jest wynik strategii default. Dodatkowe funkcje nagrody zmieniły przebieg optymalizacji, ale nie dały jednoznacznej poprawy względem oryginalnej nagrody LunarLandera. Eksperyment pokazuje też, że reward shaping trzeba projektować ostrożnie, bo dodatkowe kary i bonusy mogą zmienić skalę fitnessu i utrudnić bezpośrednie porównywanie wyników.