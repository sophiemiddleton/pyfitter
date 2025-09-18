import pickle

def WritePkl(my_data):
  # Specify the filename for your pickle file
  filename = "my_data_list.pkl"

  # Save the list to the .pkl file
  try:
      with open(filename, 'wb') as file:
          pickle.dump(my_data, file)
      print(f"List successfully saved to {filename}")
  except Exception as e:
      print(f"Error saving list: {e}")
      
def ReadPkl(filename, my_data):
  # To confirm it worked, you can load the data back:
  loaded_data = None
  try:
      with open(filename, 'rb') as file:
          loaded_data = pickle.load(file)
      print(f"\nList successfully loaded from {filename}:")
      print(loaded_data)
      print(f"Type of loaded data: {type(loaded_data)}")
      print(f"Is loaded_data equal to my_data? {loaded_data == my_data}")
  except Exception as e:
      print(f"Error loading list: {e}")
    
def main():
  # Your list of data
  my_data = [
      "This is a string",
      12345,
      {"key": "value", "number": 42},
      [1, 2, 3, 4, 5],
      True,
      None
  ]
  WritePkl(my_data)
  ReadPkl('my_data_list.pkl',my_data)
 
if __name__ == "__main__":
  main()
