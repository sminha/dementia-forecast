// import RNFS from 'react-native-fs';
// import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

// export async function extractSleepData(files: RNFS.ReadDirItem[]) {
//   const summaryFile = files.find(file => file.name.includes('sleep') && !file.name.includes('_') && file.name.endsWith('.csv'));
//   const sleepLogsFile = files.find(file => file.name.includes('sleep_stage') && file.name.endsWith('.csv'));
//   if (!summaryFile || !sleepLogsFile) { return null; }

//   const summaryContent = await RNFS.readFile(summaryFile.path, 'utf8');
//   const sleepLogsContent = await RNFS.readFile(sleepLogsFile.path, 'utf8');

//   const summary = extractSummaryData(summaryContent);
//   const sleep_logs = extractSleepLogsData(sleepLogsContent);

//   return { summary, sleep_logs };
// }

// function extractSummaryData(summaryContent: string) {
//   const featureLine = summaryContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['com.samsung.health.sleep.start_time',
//                            'com.samsung.health.sleep.end_time',
//                            'total_light_duration',
//                            'total_rem_duration',
//                            'sleep_duration',
//                            'efficiency'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLine = summaryContent.trim().split('\n')[2];
//   const data = dataLine.split(',');

//   const start_time = convertToKSTFromUTC(data[indices[0]]);
//   const end_time = convertToKSTFromUTC(data[indices[1]]);
//   const total_light_duration = parseInt(data[indices[2]], 10);
//   const total_rem_duration = parseInt(data[indices[3]], 10);
//   const total_duration = parseInt(data[indices[4]], 10);
//   const efficiency = parseInt(data[indices[5]], 10);
//   const total_deep_duration = total_duration - total_light_duration - total_rem_duration;

//   return { start_time, end_time, total_light_duration, total_deep_duration, total_rem_duration, total_duration, efficiency };
// }

// function extractSleepLogsData(sleepLogsContent: string) {
//   const featureLine = sleepLogsContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['datauuid', 'stage', 'start_time', 'end_time'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = sleepLogsContent.trim().split('\n').slice(2);
//   // const extractedData = dataLines.map(line => {
//   let extractedData = dataLines.map(line => { // FOR TEST
//     const cols = line.split(',');
//     return indices.map(i => cols[i]);
//   });

//   extractedData = extractedData.slice(0, 5); // FOR TEST
//   // console.log(extractedData);

//   // return extractedData;

//   const parsedData = extractedData.map(row => ({
//     id: row[0],
//     stage: parseInt(row[1], 10),
//     start_time: convertToKSTFromUTC(row[2]),
//     end_time: convertToKSTFromUTC(row[3]),
//   }));

//   // console.log(parsedData);

//   return parsedData;
// }

import RNFS from 'react-native-fs';
import { convertToKSTFromUTC } from '../../utils/timeUtils.ts';

export async function extractSleepData(files: RNFS.ReadDirItem[]) {
  const summaryFile = files.find(file => file.name.includes('sleep') && !file.name.includes('_') && file.name.endsWith('.csv'));
  const sleepLogsFile = files.find(file => file.name.includes('sleep_stage') && file.name.endsWith('.csv'));
  if (!summaryFile || !sleepLogsFile) { return {}; }

  const summaryContent = await RNFS.readFile(summaryFile.path, 'utf8');
  const sleepLogsContent = await RNFS.readFile(sleepLogsFile.path, 'utf8');

  const summary = extractSummaryData(summaryContent);
  const sleep_logs = extractSleepLogsData(sleepLogsContent);

  return { summary, sleep_logs };
}

function extractSummaryData(summaryContent: string) {
  const lines = summaryContent.trim().split('\n');
  const features = lines[1].split(',');

  const desiredFeatures = [
    'com.samsung.health.sleep.start_time',
    'com.samsung.health.sleep.end_time',
    'total_light_duration',
    'total_rem_duration',
    'sleep_duration',
    'efficiency',
  ];
  const indices = desiredFeatures.map(feature => features.indexOf(feature));

  const summaries = lines.slice(2).map(line => {
    const data = line.split(',');

    const start_time = convertToKSTFromUTC(data[indices[0]]);
    const end_time = convertToKSTFromUTC(data[indices[1]]);
    const total_light_duration = parseInt(data[indices[2]], 10);
    const total_rem_duration = parseInt(data[indices[3]], 10);
    const total_duration = parseInt(data[indices[4]], 10);
    const efficiency = parseInt(data[indices[5]], 10);
    const total_deep_duration = total_duration - total_light_duration - total_rem_duration;

    return { start_time, end_time, total_light_duration, total_deep_duration, total_rem_duration, total_duration, efficiency };
  });

  return summaries;
}

function extractSleepLogsData(sleepLogsContent: string) {
  const featureLine = sleepLogsContent.split('\n')[1];
  const features = featureLine.split(',');

  const desiredFeatures = ['datauuid', 'stage', 'start_time', 'end_time'];
  const indices = desiredFeatures.map(feature => features.indexOf(feature));

  const dataLines = sleepLogsContent.trim().split('\n').slice(2);
  let extractedData = dataLines.map(line => {
    const cols = line.split(',');
    return indices.map(i => cols[i]);
  });

  // extractedData = extractedData.slice(0, 5); // FOR TEST

  const parsed = extractedData.map(row => {
    const start = convertToKSTFromUTC(row[2]);
    const date = start.split('T')[0];
    return {
      date,
      data: {
        id: row[0],
        stage: parseInt(row[1], 10),
        start_time: start,
        end_time: convertToKSTFromUTC(row[3]),
      }
    };
  });

  const grouped: Record<string, any[]> = {};
  parsed.forEach(item => {
    if (!grouped[item.date]) grouped[item.date] = [];
    grouped[item.date].push(item.data);
  });

  return grouped;
}