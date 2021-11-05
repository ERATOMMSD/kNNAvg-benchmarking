echo "#####################################"
echo "Starting synchronisation with cluster"
cd ..

# Copy sources to all servers
for i in {0,1,2,3}; do \
  echo "------ Sync to aloha0$i -------"
  rsync -auvz \
    --exclude '__pycache__' \
    --exclude '*venv*' \
    --exclude '*output*' \
    --exclude '.idea' \
    --exclude '.git' \
    ./2021-08_full_benchmarks \
    aloha0$i:~/SBT/; \
  rsync -auvz \
    --exclude '*egg-info' \
    --exclude '__pycache__' \
    --exclude '.idea' \
    --exclude '*.pyc' \
    --exclude '.git' \
    ./noisy-moo \
    aloha0$i:~/SBT/; done

echo "#####################################"
echo "Finished synchronisation with cluster"
echo "\n\n"
