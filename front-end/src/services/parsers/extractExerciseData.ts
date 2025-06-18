// import RNFS from 'react-native-fs';
// import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

// export async function extractExerciseData(files: RNFS.ReadDirItem[]) {
//   const target = files.find(file => file.name.includes('exercise') && !file.name.includes('recovery_heart_rate') && file.name.endsWith('.csv'));
//   if (!target) { return null; }

//   const content = await RNFS.readFile(target.path, 'utf8');

//   const featureLine = content.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['com.samsung.health.exercise.exercise_type',
//                            'com.samsung.health.exercise.start_time',
//                            'com.samsung.health.exercise.end_time',
//                            'com.samsung.health.exercise.duration',
//                            'com.samsung.health.exercise.mean_heart_rate',
//                            'com.samsung.health.exercise.calorie'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = content.trim().split('\n').slice(2);
//   // const extractedData = dataLines.map(line => {
//   let extractedData = dataLines.map(line => { // FOR TEST
//     const cols = line.split(',');
//     return indices.map(i => cols[i]);
//   });

//   extractedData = extractedData.slice(0, 5); // FOR TEST
//   // console.log(extractedData);

//   // return extractedData;

//   const parsedData = extractedData.map(row => ({
//     type: parseInt(row[0], 10),
//     start_time: convertToKSTFromUTC(row[1]),
//     end_time: convertToKSTFromUTC(row[2]),
//     duration: parseInt(row[3], 10),
//     average_heart_rate: parseInt(row[4], 10),
//     calorie: parseFloat(row[5]),
//   }));

//   // console.log(parsedData);

//   return parsedData;
// }

import RNFS from 'react-native-fs';
import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

export async function extractExerciseData(files: RNFS.ReadDirItem[]) {
  const target = files.find(file => file.name.includes('exercise') && !file.name.includes('recovery_heart_rate') && !file.name.includes('periodization') && file.name.endsWith('.csv'));
  if (!target) return {};

  const content = await RNFS.readFile(target.path, 'utf8');
  const lines = content.trim().split('\n');
  const features = lines[1].split(',');

  const desiredFeatures = [
    'com.samsung.health.exercise.exercise_type',
    'com.samsung.health.exercise.start_time',
    'com.samsung.health.exercise.end_time',
    'com.samsung.health.exercise.duration',
    'com.samsung.health.exercise.mean_heart_rate',
    'com.samsung.health.exercise.calorie',
  ];
  const indices = desiredFeatures.map(f => features.indexOf(f));
  const dataLines = lines.slice(2);

  const parsed = dataLines.map(line => {
    const cols = line.split(',');
    const [start, end] = [convertToKSTFromUTC(cols[indices[1]]), convertToKSTFromUTC(cols[indices[2]])];
    const date = start.split('T')[0];
    return {
      date,
      data: {
        type: parseInt(cols[indices[0]], 10),
        start_time: start,
        end_time: end,
        duration: parseInt(cols[indices[3]], 10),
        average_heart_rate: parseInt(cols[indices[4]], 10),
        calorie: parseFloat(cols[indices[5]])
      }
    };
  });

  const grouped: Record<string, any[]> = {};
  parsed.forEach(item => {
    if (!grouped[item.date]) { grouped[item.date] = []; }
    grouped[item.date].push(item.data);
  });

  return grouped;
}