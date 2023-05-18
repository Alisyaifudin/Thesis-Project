with open(f"mcmc-mond-full.sh", "w") as file:
    for i in range(13):
        file.write(f"python mcmc-mond-full-script.py {i} 50000 16 False | tee logs/mond-{i} &&\n")