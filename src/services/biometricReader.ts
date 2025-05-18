import RNFS from 'react-native-fs';
import { extractExerciseData } from './parsers/extractExerciseData.ts';
import { extractWalkData } from './parsers/extractWalkData.ts';
import { extractSleepData } from './parsers/extractSleepData.ts';
import { extractHeartRateData } from './parsers/extractHeartRateData.ts';

export async function biometricReader(folderpath: string) {
  const files = await RNFS.readDir(folderpath);

  const results = {
    biometric_data_list: [
      {
        biometric_data_type: 'exercise',
        biometric_data_value: await extractExerciseData(files),
      },
      {
        biometric_data_type: 'walk',
        biometric_data_value: await extractWalkData(files),
      },
      {
        biometric_data_type: 'sleep',
        biometric_data_value: await extractSleepData(files),
      },
      {
        biometric_data_type: 'heart_rate',
        biometric_data_value: await extractHeartRateData(files, folderpath),
      },
    ],
    // exericse: await extractExerciseData(files),
    // walk: await extractWalkData(files),
    // sleep: await extractSleepData(files),
    // heart_rate: await extractHeartRateData(files, folderpath),
  };

  return results;
}