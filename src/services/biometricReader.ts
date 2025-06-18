import RNFS from 'react-native-fs';
import { extractExerciseData } from './parsers/extractExerciseData.ts';
import { extractWalkData } from './parsers/extractWalkData.ts';
import { extractSleepData } from './parsers/extractSleepData.ts';
import { extractHeartRateData } from './parsers/extractHeartRateData.ts';

function extractDateOnly(dateTimeString: string) {
  return dateTimeString.split(' ')[0];
}

function getDataByDate<T>(data: Record<string, T[]> | T[], date: string): T[] {
  if (Array.isArray(data)) {
    return data.filter((item: any) => item.start_time?.startsWith?.(date));
  }

  const matchedKeys = Object.keys(data).filter(key => key.startsWith(date));
  return matchedKeys.reduce((acc: T[], key) => acc.concat(data[key]), []);
}

function getSleepDataByDate<T>(data: Record<string, T[]> | T[], date: string): T[] {
  if (Array.isArray(data)) {
    return data.filter((item: any) => item.end_time?.startsWith?.(date));
  }

  const matchedKeys = Object.keys(data).filter(key => key.startsWith(date));
  return matchedKeys.reduce((acc: T[], key) => acc.concat(data[key]), []);
}

function calculateWalkSummary(stepLogs: any[]): { total_step_count: number; total_distance: number } {
  const total_step_count = stepLogs.reduce((acc, log) => acc + log.step_count, 0);
  const total_distance = Number(stepLogs.reduce((acc, log) => acc + log.distance, 0).toFixed(2));
  return { total_step_count, total_distance };
}

function calculateHeartRateSummary(logs: any[]): { average_heart_rate: number } {
  const total = logs.reduce((acc, log) => acc + (log.heart_rate || 0), 0);
  const average = logs.length > 0 ? Number((total / logs.length).toFixed(2)) : 0;
  return { average_heart_rate: average };
}

export async function biometricReader(folderpath: string) {
  const files = await RNFS.readDir(folderpath);

  const exerciseMap = await extractExerciseData(files);
  const walkData = await extractWalkData(files);
  const sleepData = await extractSleepData(files);
  const heartRateData = await extractHeartRateData(files, folderpath);

  const exerciseDates = Object.keys(exerciseMap).map(extractDateOnly);
  const walkDates = Object.keys(walkData.step_logs || {}).map(extractDateOnly);
  const sleepDates = Object.keys(sleepData.sleep_logs || {}).map(extractDateOnly);
  const heartRateDates = Object.keys(heartRateData.heart_rate_logs || {}).map(extractDateOnly);

  const allDates = [exerciseDates, walkDates, sleepDates, heartRateDates];
  const commonDates = allDates.reduce((a, b) => a.filter(d => b.includes(d)));

  const sortedDates = [...new Set(commonDates)].sort();
  const now = new Date();
  const twoWeeksAgo = new Date(now);
  // twoWeeksAgo.setDate(now.getDate() - 60); // FOR TEST - 60일 이내
  twoWeeksAgo.setDate(now.getDate() - 21);

  // FOR TEST - 연속 2일치
  // const continuousPairs: [string, string][] = [];
  // for (let i = 0; i < sortedDates.length - 1; i++) {
  //   const d1 = new Date(sortedDates[i]);
  //   const d2 = new Date(sortedDates[i + 1]);
  //   const diffDays = (d2.getTime() - d1.getTime()) / (1000 * 60 * 60 * 24);

  //   if (diffDays === 1 && d1 >= twoWeeksAgo && d2 >= twoWeeksAgo) {
  //     continuousPairs.push([sortedDates[i], sortedDates[i + 1]]);
  //   }
  // }

  // const latestPair = continuousPairs.length > 0
  //   ? continuousPairs.reduce((latest, current) =>
  //       new Date(current[1]) > new Date(latest[1]) ? current : latest
  //     )
  //   : null;

  // const uniqueDates = latestPair ? [...new Set(latestPair)] : [];

  // 연속 3일치
  const continuousTriplets: [string, string, string][] = [];
  for (let i = 0; i < sortedDates.length - 2; i++) {
    const d1 = new Date(sortedDates[i]);
    const d2 = new Date(sortedDates[i + 1]);
    const d3 = new Date(sortedDates[i + 2]);

    const diff1 = (d2.getTime() - d1.getTime()) / (1000 * 60 * 60 * 24);
    const diff2 = (d3.getTime() - d2.getTime()) / (1000 * 60 * 60 * 24);

    if (diff1 === 1 && diff2 === 1 && d1 >= twoWeeksAgo && d3 >= twoWeeksAgo) {
      continuousTriplets.push([sortedDates[i], sortedDates[i + 1], sortedDates[i + 2]]);
    }
  }

  const latestTriplet = continuousTriplets.length > 0
    ? continuousTriplets.reduce((latest, current) =>
        new Date(current[2]) > new Date(latest[2]) ? current : latest
      )
    : null;

  const uniqueDates = latestTriplet ? [...new Set(latestTriplet)] : [];

  const biometric_data_by_date = uniqueDates.map(date => ({
    date,
    biometric_data_list: [
      {
        biometric_data_type: 'exercise',
        biometric_data_value: getDataByDate(exerciseMap, date),
      },
      {
        biometric_data_type: 'walk',
        biometric_data_value: walkData.step_logs
          ? getDataByDate(walkData.step_logs, date).length > 0
            ? {
                summary: calculateWalkSummary(getDataByDate(walkData.step_logs, date)),
                step_logs: getDataByDate(walkData.step_logs, date),
              }
            : null
          : null,
      },
      {
        biometric_data_type: 'sleep',
        biometric_data_value: sleepData.sleep_logs
          ? getDataByDate(sleepData.sleep_logs, date).length > 0
            ? {
                summary: sleepData.summary ? getSleepDataByDate(sleepData.summary, date)[0] || null : null,
                sleep_logs: getDataByDate(sleepData.sleep_logs, date),
              }
            : null
          : null,
      },
      {
        biometric_data_type: 'heart_rate',
        biometric_data_value: heartRateData.heart_rate_logs
          ? getDataByDate(heartRateData.heart_rate_logs, date).length > 0
            ? {
                summary: calculateHeartRateSummary(
                  getDataByDate(heartRateData.heart_rate_logs, uniqueDates[0]).concat(
                    getDataByDate(heartRateData.heart_rate_logs, uniqueDates[1])
                  )
                ),
                heart_rate_logs: getDataByDate(heartRateData.heart_rate_logs, date),
                rmssd_logs: heartRateData.rmssd_logs
                  ? getDataByDate(heartRateData.rmssd_logs, date)
                  : [],
              }
            : null
          : null,
      },
    ],
  }));

  return { biometric_data_by_date };
}