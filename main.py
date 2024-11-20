import subprocess

# Start both scripts in parallel
process1 = subprocess.Popen(["python", "camera1.py"])
process2 = subprocess.Popen(["python", "camera2.py"])

# Wait for both scripts to finish
process1.wait()
process2.wait()

print("Both camera scripts have finished executing.")

process3 = subprocess.Popen(["python", "trainerModel.py"])

process3.wait()

choice = input("Would you like to open the facial recognition to test it? (y/n)")

if choice == "y":
    process4 = subprocess.Popen(["python", "facialRecog.py"])
    process4.wait()


print("All done then thanks. :)")