// import RNFS from 'react-native-fs';
// import { convertToKSTFromUTC, convertToKSTFromTimestamp } from '../../utils/timeUtils.ts';

// export async function extractHeartRateData(files: RNFS.ReadDirItem[], folderPath: string) {
//   const heartRateLogsFile = files.find(file => file.name.includes('heart_rate') && !file.name.includes('alerted_') && !file.name.includes('recovery_') && file.name.endsWith('.csv'));
//   const rmssdLogsFile = files.find(file => file.name.includes('hrv') && file.name.endsWith('.csv'));
//   if (!heartRateLogsFile || !rmssdLogsFile) { return null; }

//   const heartRateLogsContent = await RNFS.readFile(heartRateLogsFile.path, 'utf8');
//   const rmssdLogsContent = await RNFS.readFile(rmssdLogsFile.path, 'utf8');

//   const summary = await extractSummaryData(heartRateLogsContent);
//   const heart_rate_logs = await extractHeartRateLogsData(heartRateLogsContent);
//   const rmssd_logs = await extractRmssdLogsData(rmssdLogsContent, folderPath);

//   return { summary, heart_rate_logs, rmssd_logs };
// }

// function extractSummaryData(heartRateLogsContent: string) {
//   const featureLine = heartRateLogsContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['com.samsung.health.heart_rate.heart_rate'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = heartRateLogsContent.trim().split('\n').slice(2);
//   let total_heart_rate = 0;

//   dataLines.forEach(line => {
//     const cols = line.split(',');

//     const heart_rate = parseInt(cols[indices[0]], 10);

//     total_heart_rate += heart_rate;
//   });

//   const average_heart_rate = Number((total_heart_rate / dataLines.length).toFixed(2));

//   // console.log(total_heart_rate);
//   // console.log(average_heart_rate);

//   return { average_heart_rate };
// }

// function extractHeartRateLogsData(heartRateLogsContent: string) {
//   const featureLine = heartRateLogsContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['com.samsung.health.heart_rate.start_time',
//                            'com.samsung.health.heart_rate.end_time',
//                            'com.samsung.health.heart_rate.heart_rate',];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = heartRateLogsContent.trim().split('\n').slice(2);
//   // const extractedData = dataLines.map(line => {
//   let extractedData = dataLines.map(line => { // FOR TEST
//     const cols = line.split(',');
//     return indices.map(i => cols[i]);
//   });

//   extractedData = extractedData.slice(0, 5); // FOR TEST
//   // console.log(extractedData);

//   // return extractedData;

//   const parsedData = extractedData.map(row => ({
//     start_time: convertToKSTFromUTC(row[0]),
//     end_time: convertToKSTFromUTC(row[1]),
//     heart_rate: parseInt(row[2], 10),
//   }));

//   // console.log(parsedData);

//   return parsedData;
// }

// async function extractRmssdLogsData(rmssdLogsContent: string, folderPath: string) {
//   const featureLine = rmssdLogsContent.split('\n')[1];
//   const features = featureLine.split(',');

//   const desiredFeatures = ['start_time', 'binning_data'];
//   const indices = desiredFeatures.map(feature => features.indexOf(feature));

//   const dataLines = rmssdLogsContent.trim().split('\n').slice(2);
//   // const extractedData = dataLines.map(line => {
//   let extractedData = dataLines.map(line => { // FOR TEST
//     const cols = line.split(',');
//     return indices.map(i => cols[i]);
//   });

//   extractedData = extractedData.slice(0, 5); // FOR TEST
//   // console.log(extractedData);

//   // return extractedData;

//   const rmssdFolderPath = `${folderPath}/jsons/com.samsung.health.hrv`;
//   const exists = await RNFS.exists(rmssdFolderPath);
//   if (!exists) {
//     console.log('하위 폴더가 없습니다.');
//     return;
//   }

//   let jsonData = [];
//   // const jsonResults = [];

//   for (const [startTime, binningData] of extractedData) {
//     const firstLetter = binningData[0];
//     const rmssdJsonPath = `${rmssdFolderPath}/${firstLetter}/${binningData}`;

//     const jsonContent = await RNFS.readFile(rmssdJsonPath, 'utf8');
//     jsonData = JSON.parse(jsonContent);

//     jsonData.forEach((entry: any) => {
//       entry.start_time = convertToKSTFromTimestamp(entry.start_time);
//       entry.end_time = convertToKSTFromTimestamp(entry.end_time);
//       delete entry.sdnn;
//       entry.rmssd = Number(entry.rmssd.toFixed(2));
//     });

//     // const firstLetter = binningData[0];
//     // const rmssdJsonPath = `${rmssdFolderPath}/${firstLetter}/${binningData}`;

//     // try {
//     //   const jsonContent = await RNFS.readFile(rmssdJsonPath, 'utf8');
//     //   const jsonData = JSON.parse(jsonContent);
//     //   jsonData.forEach(entry => {
//     //     delete entry.sdnn;
//     //   });
//     //   jsonResults.push(jsonData);
//     // } catch (error) {
//     //   console.log(`JSON 파일 읽기 실패: ${rmssdJsonPath}`, error);
//     //   jsonResults.push({ startTime, jsonData: null });
//     // }
//   }

//   jsonData = jsonData.slice(0, 5); // FOR TEST
//   // console.log(jsonData);

//   return jsonData;
//   // return jsonResults;
// }

import RNFS from 'react-native-fs';
import { convertToKSTFromUTC, convertToKSTFromTimestamp } from '../../utils/timeUtils.ts';

export async function extractHeartRateData(files: RNFS.ReadDirItem[], folderPath: string) {
  const heartRateLogsFile = files.find(file => file.name.includes('heart_rate') && !file.name.includes('alerted_') && !file.name.includes('recovery_') && file.name.endsWith('.csv'));
  const rmssdLogsFile = files.find(file => file.name.includes('hrv') && file.name.endsWith('.csv'));
  if (!heartRateLogsFile || !rmssdLogsFile) { return {}; }

  const heartRateLogsContent = await RNFS.readFile(heartRateLogsFile.path, 'utf8');
  const rmssdLogsContent = await RNFS.readFile(rmssdLogsFile.path, 'utf8');

  const heart_rate_logs = extractHeartRateLogsData(heartRateLogsContent);
  const rmssd_logs = await extractRmssdLogsData(rmssdLogsContent, folderPath);

  return { heart_rate_logs, rmssd_logs };
}

function extractHeartRateLogsData(heartRateLogsContent: string) {
  const featureLine = heartRateLogsContent.split('\n')[1];
  const features = featureLine.split(',');

  const desiredFeatures = [
    'com.samsung.health.heart_rate.start_time',
    'com.samsung.health.heart_rate.end_time',
    'com.samsung.health.heart_rate.heart_rate',
  ];
  const indices = desiredFeatures.map(feature => features.indexOf(feature));

  const dataLines = heartRateLogsContent.trim().split('\n').slice(2);
  let extractedData = dataLines.map(line => {
    const cols = line.split(',');
    return indices.map(i => cols[i]);
  });

  // extractedData = extractedData.slice(0, 5); // FOR TEST

  const parsed = extractedData.map(row => {
    const start = convertToKSTFromUTC(row[0]);
    const end = convertToKSTFromUTC(row[1]);
    const date = start.split('T')[0];
    return {
      date,
      data: {
        start_time: start,
        end_time: end,
        heart_rate: parseInt(row[2], 10),
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

async function extractRmssdLogsData(rmssdLogsContent: string, folderPath: string) {
  const featureLine = rmssdLogsContent.split('\n')[1];
  const features = featureLine.split(',');

  const desiredFeatures = ['start_time', 'binning_data'];
  const indices = desiredFeatures.map(feature => features.indexOf(feature));

  const dataLines = rmssdLogsContent.trim().split('\n').slice(2);
  let extractedData = dataLines.map(line => {
    const cols = line.split(',');
    return indices.map(i => cols[i]);
  });

  // extractedData = extractedData.slice(0, 5); // FOR TEST

  const rmssdFolderPath = `${folderPath}/jsons/com.samsung.health.hrv`;
  const exists = await RNFS.exists(rmssdFolderPath);
  if (!exists) {
    console.log('하위 폴더가 없습니다.');
    return {};
  }

  const allData: { date: string; data: any }[] = [];

  for (const [startTime, binningData] of extractedData) {
    const firstLetter = binningData[0];
    const rmssdJsonPath = `${rmssdFolderPath}/${firstLetter}/${binningData}`;

    try {
      const jsonContent = await RNFS.readFile(rmssdJsonPath, 'utf8');
      const jsonData = JSON.parse(jsonContent);

      jsonData.forEach((entry: any) => {
        entry.start_time = convertToKSTFromTimestamp(entry.start_time);
        entry.end_time = convertToKSTFromTimestamp(entry.end_time);
        delete entry.sdnn;
        entry.rmssd = Number(entry.rmssd.toFixed(2));

        const date = entry.start_time.split('T')[0];
        allData.push({ date, data: entry });
      });
    } catch (error) {
      console.log(`JSON 파일 읽기 실패: ${rmssdJsonPath}`, error);
    }
  }

  const grouped: Record<string, any[]> = {};
  allData.forEach(item => {
    if (!grouped[item.date]) grouped[item.date] = [];
    grouped[item.date].push(item.data);
  });

  return grouped;
}