# Ewolucja sieci neuronowych w środowisku Lunar Lander (NEAT)

## Opis i Cel Projektu

Projekt badawczy poświęcony jest analizie procesów neuroewolucyjnych z wykorzystaniem algorytmu **NEAT (NeuroEvolution of Augmenting Topologies)**. Głównym celem jest wyhodowanie autonomicznego agenta (lądownika), który nauczy się bezpiecznego i precyzyjnego lądowania na nieregularnym podłożu w symulacji **Lunar Lander** (dostarczanej przez bibliotekę *Gymnasium*).

W przeciwieństwie do klasycznych metod uczenia ze wzmocnieniem, gdzie architektura sieci neuronowej jest stała, algorytm NEAT pozwala na równoczesną ewolucję wag synaptycznych oraz **struktury (topologii) sieci** — zaczynając od form minimalnych i stopniowo je komplikując w drodze doboru naturalnego.

---

## Kluczowe Założenia i Możliwości Badawcze

### 1. Inżynieria Funkcji Przystosowania (Fitness Function Engineering)

Sercem projektu jest badanie wpływu kryteriów oceny na zachowanie agentów. Architektura pozwala na dynamiczne podmienianie strategii nagradzania. W projekcie zaimplementowano pięć strategii funkcji przystosowania:

| Strategia | Identyfikator | Logika |
|---|---|---|
| **Domyślna** | `"default"` | Natywny system nagród środowiska Gymnasium |
| **Presja czasu** | `"penalty_time"` | Stała kara `−0.2` za każdy krok symulacji, zachęcająca do szybszego zakończenia epizodu |
| **Stabilizacja kąta** | `"penalty_angle"` | Kara za przechylenie lądownika w momencie kontaktu z podłożem |
| **Jakość lądowania** | `"landing_quality"` | Kara za odległość od środka, prędkość i przechylenie oraz bonus za stabilny kontakt obu nóg |
| **Centrowanie lądowania** | `"centered_landing"` | Silniejsza kara za oddalenie od środka lądowiska |

Zmiana strategii nie wymaga modyfikacji kodu — wystarczy edytować pole `fitness_strategy` w pliku `config/simulation_config.json`.

### 2. Kontrola Warunków Środowiskowych (Generalizacja vs Overfitting)

Aby ocena agentów była sprawiedliwa i miarodajna, system umożliwia testowanie każdej sieci neuronowej na przestrzeni kilku niezależnych prób w ramach jednej generacji. W zależności od konfiguracji eksperymentu można badać:

- **Zdolność do zapamiętywania (seedy stałe):** Agenci są testowani na tych samych, z góry zdefiniowanych układach terenu, co pozwala precyzyjnie porównywać ich efektywność w znanym świecie.
- **Zdolność do generalizacji (seedy zmienne):** Ukształtowanie terenu zmienia się dynamicznie z generacji na generację, co zmusza populację do wypracowania uniwersalnych odruchów i zapobiega „wykuciu się mapy na pamięć".

Parametr `use_fixed_seeds` w pliku konfiguracyjnym przełącza między tymi trybami.

### 3. Skalowalność i Analiza Postępów

Proces ewolucji wspiera pełne przetwarzanie równoległe (`ParallelEvaluator`), dzięki czemu setki osobników mogą przechodzić testy jednocześnie na wszystkich dostępnych rdzeniach procesora. Dla celów demonstracyjnych i analitycznych system pozwala na podgląd na żywo zachowania „najlepszego osobnika" z danej generacji. Dodatkowo proces nauki jest zabezpieczony automatycznymi punktami zapisu (checkpointami), co pozwala na długofalowe eksperymenty, zatrzymywanie i wznawianie symulacji bez utraty wyewoluowanej puli genowej.

---

## Architektura Projektu

```
project/
├── config/
│   ├── neat_config.txt          # parametry algorytmu NEAT (rozmiar populacji, mutacje, speciacja)
│   └── simulation_config.json   # flagi eksperymentu (strategia, seedy, render, checkpointy)
├── src/
│   ├── fitness_functions.py     # 5 strategii funkcji przystosowania + słownik FITNESS_STRATEGIES
│   ├── experiment_logger.py     # zapis konfiguracji i statystyk eksperymentu
│   ├── simulation.py            # klasa LunarSimulation — pętla epizodu Gymnasium
│   └── visualization.py        # wykresy postępu treningu i topologii sieci
├── scripts/
│   └── summarize_results.py     # generuje summary.csv i wykres porównawczy
├── results/
│   └── <fitness_strategy>/      # wyniki osobnych eksperymentów
├── EXPERIMENTS.md               # szczegółowa analiza wyników
├── main.py                      # orchestracja: NEAT loop, ParallelEvaluator, checkpointy
└── requirements.txt
```

### Przepływ danych w jednej generacji

```
Population (150 genomów)
    │
    ▼  [równolegle na N rdzeniach CPU]
ParallelEvaluator
    │   dla każdego genomu:
    │     FeedForwardNetwork.create(genome, config)
    │     LunarSimulation.run_agent(net)  ← 3 epizody × max 500 kroków
    │       env.reset(seed) → pętla: net.activate(obs) → argmax → env.step()
    │       fitness += strategy_func(reward, info, step, obs)
    │     return mean(fitness over 3 runs)
    │
    ▼
Selekcja + Speciacja + Reprodukcja (NEAT)
    │
    ▼  [co 10 generacji]
Checkpoint zapisany → neat-checkpoint-N
```

---

## Przestrzeń Stanów i Akcji

Agent obserwuje 8 wartości ciągłych opisujących stan lądownika:

| Indeks | Obserwacja | Opis |
|---|---|---|
| 0 | `x_pos` | Pozycja pozioma |
| 1 | `y_pos` | Pozycja pionowa |
| 2 | `x_vel` | Prędkość pozioma |
| 3 | `y_vel` | Prędkość pionowa |
| 4 | `angle` | Kąt przechylenia (rad) |
| 5 | `ang_vel` | Prędkość kątowa |
| 6 | `leg_L` | Kontakt lewej nogi z podłożem (0/1) |
| 7 | `leg_R` | Kontakt prawej nogi z podłożem (0/1) |

Na podstawie tych obserwacji sieć wybiera jedną z 4 dyskretnych akcji:

| Wyjście | Akcja |
|---|---|
| 0 | Nic (swobodny lot) |
| 1 | Lewy silnik boczny |
| 2 | Główny silnik (dół→góra) |
| 3 | Prawy silnik boczny |

Wybór akcji: `action = argmax(net.activate(observation))`.

---

## Konfiguracja NEAT (`config/neat_config.txt`)

Kluczowe parametry algorytmu:

| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `pop_size` | 150 | Wystarczająca różnorodność bez nadmiernego kosztu obliczeniowego |
| `fitness_threshold` | 250.0 | Próg referencyjny powyżej progu „rozwiązanego" środowiska. Przy `no_fitness_termination = True` nie zatrzymuje eksperymentu wcześniej |
| `num_hidden` | 0 | Start z minimalną siecią — NEAT sam dodaje węzły |
| `initial_connection` | `full_direct` | Pełne połączenie wejść z wyjściami na start |
| `activation_default` | `tanh` | Zablokowane na tanh (stabilne gradienty, wyjście w \[−1, 1\]) |
| `elitism` | 2 | Top 2 genomy każdego gatunku przechodzą bez mutacji |
| `max_stagnation` | 20 | Gatunek ma 20 generacji na poprawę zanim zostanie usunięty |
| `compatibility_threshold` | 3.0 | Próg dystansu genomicznego przy tworzeniu gatunków |
| `no_fitness_termination` | `True` | Wymusza pełne 300 generacji dla każdego eksperymentu, nawet jeśli zostanie osiągnięty `fitness_threshold` |

---

## Plik Konfiguracyjny Eksperymentu (`config/simulation_config.json`)

```json
{
  "simulation": {
    "max_steps_per_episode": 500,
    "num_eval_runs": 3,
    "use_fixed_seeds": true,
    "fixed_seeds": [42, 123, 999],
    "fitness_strategy": "default"
  },
  "neat_runtime": {
    "checkpoint_generation_interval": 10,
    "resume_from_latest_checkpoint": false,
    "render_best_after_generation": true
  }
}
```

| Parametr | Opis |
|---|---|
| `fitness_strategy` | `"default"` / `"penalty_time"` / `"penalty_angle"` / `"landing_quality"` / `"centered_landing"` |
| `use_fixed_seeds` | `true` → stałe seedy (porównywalność); `false` → zmienne seedy (generalizacja) |
| `fixed_seeds` | Lista seedów środowiska — jeden per próba ewaluacyjna |
| `num_eval_runs` | Liczba prób per genom (fitness = średnia) |
| `resume_from_latest_checkpoint` | `true` → wznowienie z ostatniego `neat-checkpoint-N` |
| `render_best_after_generation` | `true` → okno wizualizacji najlepszego agenta po każdej generacji |

---

## Instalacja i Uruchomienie

### Wymagania

```
neat-python >= 2.0.0
gymnasium[box2d] >= 1.0.0
matplotlib >= 3.7.0
numpy >= 1.24.0
swig
```

### Instalacja zależności

```bash
pip install swig
pip install neat-python "gymnasium[box2d]"
```

> **Uwaga (Python 3.13 / Windows):** Jeśli `gymnasium[box2d]` nie instaluje się z powodu braku skompilowanych kół dla Box2D, zainstaluj najpierw `swig` (narzędzie do generowania bindingów C), a następnie spróbuj ponownie. Alternatywnie użyj Pythona 3.11, dla którego istnieją gotowe koła binarne.

### Uruchomienie

```bash
python main.py
```

Program automatycznie:
1. Uruchamia ewolucję od generacji 0 (lub wznawia z checkpointu, jeśli `resume_from_latest_checkpoint: true`)
2. Wyświetla statystyki po każdej generacji
3. Renderuje najlepszego agenta generacji (jeśli `render_best_after_generation: true`)
4. Zapisuje checkpoint co `checkpoint_generation_interval` generacji
5. Po zakończeniu zapisuje wyniki do folderu results/<fitness_strategy>/, w tym fitness_history.png, winner_network.png, training_stats.csv i config_used.json.

### Wznowienie przerwanego treningu

W `simulation_config.json` ustaw:
```json
"resume_from_latest_checkpoint": true
```
Program odnajdzie plik `neat-checkpoint-N` z najwyższym `N` i wznowi ewolucję.

---

## Wyniki i Wizualizacje

Wyniki są zapisywane w folderze odpowiadającym użytej strategii fitnessu:

```text
results/<fitness_strategy>/
```

Po zakończeniu każdego treningu generowane są 4 pliki:

- **fitness_history.png** — wykres postępu per generacja: linia maksymalnego fitnessu, linia średniego fitnessu oraz pasmo odchylenia standardowego. Pozioma linia przerywana oznacza próg „rozwiązanego" środowiska (200 pkt).
- **winner_network.png** — topologia sieci zwycięskiego genomu: węzły wejściowe (zielone), ukryte (niebieskie), wyjściowe (łososiowe); krawędzie niebieskie = wagi dodatnie, czerwone = ujemne; grubość ∝ |waga|.
- **training_stats.csv** — tabela z wartościami best_fitness, mean_fitness i std_fitness dla każdej generacji.
- **config_used.json** — konfiguracja użyta do wygenerowania danego eksperymentu.

Do zbiorczego porównania eksperymentów służy skrypt:

```bash
python scripts/summarize_results.py
```
Tworzy on:

- **results/summary.csv** — zbiorcze podsumowanie wyników,
- **results/best_fitness_comparison.png** — wykres porównujący strategie.

Szczegółowa analiza eksperymentów znajduje się w [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Kierunki Rozwoju

System został zaprojektowany w sposób modułowy, co otwiera drogę do dalszych badań:

- **Sieci rekurencyjne:** Zamiana `neat.nn.FeedForwardNetwork` na `neat.nn.RecurrentNetwork` oraz ustawienie `feed_forward = False` w `neat_config.txt` pozwoli agentom na posiadanie pamięci krótkotrwałej — bez żadnych zmian w pozostałej architekturze.
- **Dalsze strategie nagrody:** Projekt został rozszerzony o strategie `landing_quality` i `centered_landing`, ale można dalej testować inne warianty reward shapingu, np. osobne kary za zużycie paliwa, prędkość pionową przy kontakcie z ziemią albo bonus wyłącznie za poprawne zakończenie epizodu.
- **Inne środowiska Gymnasium:** Klasa `LunarSimulation` może być zastąpiona analogiczną klasą dla dowolnego środowiska dyskretnego — jedyną zmianą jest liczba wejść/wyjść w `neat_config.txt`.
