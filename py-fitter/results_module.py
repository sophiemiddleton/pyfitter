import awkward as ak
import csv

class ResultsClass:
  def __init__(self, data, result, verbose=0):
        self.result = result
        self.data = ak.flatten(data['trksegs','mom.mag'], axis=None)
        self.verbose = verbose

  def WriteFittedData(self):
    """ Write data used in fit to csv (i,mom,time) Note: should be in format useful to BAT"""
    file_path = 'output_data.csv'

    with open(file_path , 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in self.data:
            csv_writer.writerow([item])

    if self.verbose > 0:
      print("[py-fitter/results_module/WriteFittedData] ✅ Data written to {file_path}")
    
  def WriteResult(self):
    """ Write result to csv file for safe keeping """
    file_path = 'output_fitresult.csv'
    # Open the file in write mode ('w')
    with open(file_path, 'w') as csvfile:
        csvfile.write(str(self.result))

    if self.verbose > 0:
      print("[py-fitter/results_module/WriteFittedData] ✅ Result written to {file_path}")

