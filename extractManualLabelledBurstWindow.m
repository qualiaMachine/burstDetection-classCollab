savedDataDir = '/Users/tammi/Desktop/DeepANN/git/burstDetection/processedResults';

animal = 'Resh';

load([savedDataDir filesep 'burstCoords_' animal '.mat'])

for iFile = 1:size(coordDataFileList,1)
   if isempty(coordDataFileList(iFile).burstLocC)
       continue
   end
   
   burstLocC = coordDataFileList(iFile).burstLocC;
   burstLocR = coordDataFileList(iFile).burstLocR;
   
    
end