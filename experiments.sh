######################################## EXPERIMENT 1 ########################################
# Diferent initialization in various incremental learning approaches.
##############################################################################################


inits=(1)
RNS=(0 0.1)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")
for model in "${models[@]}"; do
  for init in "${inits[@]}"; do
    for RN in "${RNS[@]}"; do
      # Run the Python script with the current combination of parameters
      python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 1 -RN "$RN"
    done
  done
done

inits=(0 3 13)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")
for init in "${inits[@]}"; do
    for RN in "${RNS[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 1 -RN "$RN"
    done
done

######################################## EXPERIMENT 2 ########################################
# Number of training epochs.
##############################################################################################

epochs=(5 10 15 20 25 30 35 40 45 50 75 100)
models=("finetune" "LKGE" "incDE")
inits=(0 1)

for epoch in "${epochs[@]}"; do
  for init in "${inits[@]}"; do
    for model in "${models[@]}"; do
      # Run the Python script with the current combination of parameters
      python main.py -dataset ENTITY100 -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch"
    done
  done
done

######################################## EXPERIMENT 3 ########################################
# Number of new entities per snapshot.
##############################################################################################


datasets=("ENTITY10" "ENTITY50" "ENTITY200" "ENTITY500" "ENTITY1000")
models=("finetune" "LKGE" "incDE")
RNS=(0 0.1 0.5 1)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for RN in "${RNS[@]}"; do
      # Run the Python script with the current combination of parameters
      python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 1 -RN "$RN"
    done
  done
done

datasets=("ENTITY10" "ENTITY50" "ENTITY200" "ENTITY500" "ENTITY1000")
models=("finetune" "LKGE" "incDE")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    # Run the Python script with the current combination of parameters
    python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 0 -RN "$RN"
  done
done



######################################## EXPERIMENT 4 ########################################
# Number of training epochs and random noise.
##############################################################################################

epochs=(0 5 10 15 20 25 30 35 40 45 50 75 100 200)
models=("LKGE")
RNS=(0 0.1)

for epoch in "${epochs[@]}"; do
  for init in "${inits[@]}"; do
    for RN in "${RNS[@]}"; do
      # Run the Python script with the current combination of parameters
      python main.py -dataset ENTITY100 -gpu 0 -lifelong_name LKGE -init 1 -incremental_epochs "$epoch"
    done
  done
done