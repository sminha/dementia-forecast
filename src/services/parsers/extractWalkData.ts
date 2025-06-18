// import RNFS from 'react-native-fs';
// import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

// export async function extractWalkData(files: RNFS.ReadDirItem[]) {
//   const summaryFile = files.find(file => file.name.includes('pedometer_day_summary') && file.name.endsWith('.csv'));
//   const stepLogsFile = files.find(file => file.name.includes('pedometer_step_count') && file.name.endsWith('.csv'));
//   if (!summaryFile || !stepLogsFile) { return null; }

//   const summaryContent = await RNFS.readFile(summaryFile.path, 'utf8');
//   const stepLogsContent = await RNFS.readFile(stepLogsFile.path, 'utf8');

//   const summary = extractSummaryData(summaryContent);
//   const step_logs = extractStepLogsData(stepLogsContent);

//   return { summary, step_logs };
// }

// function extractSummaryData(summaryContent: string) {
//   const featureLine = summaryContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['create_time', 'step_count', 'distance'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = summaryContent.trim().split('\n').slice(2);
//   let total_step_count = 0;
//   let total_distance = 0;

//   dataLines.forEach(line => {
//     const cols = line.split(',');

//     const step_count = parseInt(cols[indices[1]], 10);
//     const distance = parseFloat(cols[indices[2]]);

//     total_step_count += step_count;
//     total_distance += distance;
//   });

//   // console.log(total_step_count);
//   // console.log(total_distance);

//   total_distance = Number(total_distance.toFixed(2));

//   return { total_step_count, total_distance };
// }

// function extractStepLogsData(stepLogsContent: string) {
//   const featureLine = stepLogsContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['com.samsung.health.step_count.start_time',
//                            'com.samsung.health.step_count.end_time',
//                            'com.samsung.health.step_count.count',
//                            'com.samsung.health.step_count.distance'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = stepLogsContent.trim().split('\n').slice(2);
//   // const extractedData = dataLines.map(line => { // FOR TEST
//   let extractedData = dataLines.map(line => {
//     const cols = line.split(',');
//     return indices.map(i => cols[i]);
//   });

//   extractedData = extractedData.slice(0, 5); // FOR TEST
//   // console.log(extractedData);

//   // return extractedData;

//   const parsedData = extractedData.map(row => ({
//     start_time: convertToKSTFromUTC(row[0]),
//     end_time: convertToKSTFromUTC(row[1]),
//     step_count: parseInt(row[2], 10),
//     distance: Number(parseFloat(row[3]).toFixed(2)),
//   }));

//   // console.log(parsedData);

//   return parsedData;
// }

import RNFS from 'react-native-fs';
import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

export async function extractWalkData(files: RNFS.ReadDirItem[]) {
  const stepLogsFile = files.find(file => file.name.includes('pedometer_step_count') && file.name.endsWith('.csv'));
  if (!stepLogsFile) { return {}; }

  const stepLogsContent = await RNFS.readFile(stepLogsFile.path, 'utf8');

  const step_logs = extractStepLogsData(stepLogsContent);

  return {step_logs };
}

function extractStepLogsData(stepLogsContent: string) {
  const featureLine = stepLogsContent.split('\n')[1];
  const features = featureLine.split(',');

  const desiredFeatures = ['com.samsung.health.step_count.start_time',
                           'com.samsung.health.step_count.end_time',
                           'com.samsung.health.step_count.count',
                           'com.samsung.health.step_count.distance'];
  const indices = desiredFeatures.map(feature => features.indexOf(feature));

  const dataLines = stepLogsContent.trim().split('\n').slice(2);
  let extractedData = dataLines.map(line => {
    const cols = line.split(',');
    return indices.map(i => cols[i]);
  });

  // extractedData = extractedData.slice(0, 5); // FOR TEST

  const parsed = extractedData.map(row => {
    const start = convertToKSTFromUTC(row[0]);
    const date = start.split('T')[0];
    return {
      date,
      data: {
        start_time: start,
        end_time: convertToKSTFromUTC(row[1]),
        step_count: parseInt(row[2], 10),
        distance: Number(parseFloat(row[3]).toFixed(2)),
      },
    };
  });

  const grouped: Record<string, any[]> = {};
  parsed.forEach(item => {
    if (!grouped[item.date]) grouped[item.date] = [];
    grouped[item.date].push(item.data);
  });

  return grouped;
}