#practice script to change delimitation from NORSAR seismic event export to standard csv

#file_name = 'C:/Users/Tyler/Desktop/Project Libson/ULF Plotting/test_data/regional_10.csv'
file_name = 'C:/Users/Tyler/Downloads/teleseismic_10.csv'

# Read in the file
with open(file_name, 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace(' ', ',')
filedata = filedata.replace('Origin', 'Date')

# Write the file out again
with open(file_name, 'w') as file:
  file.write(filedata)
