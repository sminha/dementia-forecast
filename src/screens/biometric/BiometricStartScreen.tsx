import React, { useRef, useEffect, useState } from 'react';
import { AppState, AppStateStatus, Linking, View, TouchableOpacity, StyleSheet } from 'react-native';
import Modal from 'react-native-modal';
import { useDispatch } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { setBiometricInfo } from '../../redux/slices/biometricSlice.ts';
import { AppDispatch } from '../../redux/store.ts';
import RNFS from 'react-native-fs';
import FastImage from 'react-native-fast-image';
import CustomText from '../../components/CustomText.tsx';
import { biometricReader } from '../../services/biometricReader.ts';
import testResults from '../../constants/test_biometric_data.json';

// const saveResultsToFile = async (results: any) => {
//   const path = `${RNFS.DownloadDirectoryPath}/mock_biometric_data.json`;
//   try {
//     await RNFS.writeFile(path, JSON.stringify(results), 'utf8');
//     console.log('결과 저장 성공:', path);
//   } catch (e) {
//     console.log('저장 실패:', e);
//   }
// };

const TWO_WEEKS_MS = 14 * 24 * 60 * 60 * 1000;

const BiometricStartScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'BiometricStart'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();

  const [modal, setModal] = useState<boolean>(false);
  const [isThreeDay, setIsThreeDay] = useState<boolean>(true);
  const [modalTitle, setModalTitle] = useState<string>('삼성헬스에서 데이터를 다운로드해주세요.');
  const [modalText, setModalText] = useState<string>('우측 상단 점 세 개-설정-개인 데이터 다운로드');

  const appState = useRef<AppStateStatus>(AppState.currentState);

  const openSamsungHealth = async () => {
    const url = 'https://apps.samsung.com/appquery/appDetail.as?appId=com.sec.android.app.shealth';

    Linking.openURL(url).catch(err => {
      console.log('Galaxy Store 열기 실패:', err);
    });
  };

  const parseDateFromFolderName = (folderName: string): Date | null => {
    const parts = folderName.split('_');
    const dateStr = parts[parts.length - 1];

    if (/^\d{14,}$/.test(dateStr)) {
      const year = parseInt(dateStr.slice(0, 4), 10);
      const month = parseInt(dateStr.slice(4, 6), 10) - 1;
      const day = parseInt(dateStr.slice(6, 8), 10);
      const hour = parseInt(dateStr.slice(8, 10), 10);
      const minute = parseInt(dateStr.slice(10, 12), 10);
      const second = parseInt(dateStr.slice(12, 14), 10);

      return new Date(year, month, day, hour, minute, second);
    }

    return null;
  };

  const readBiometricData = async () => {
    console.log('파일 읽기 시작');

    const samsungHealthPath = `${RNFS.DownloadDirectoryPath}/Samsung Health`;

    try {
      const subFolders = await RNFS.readDir(samsungHealthPath);
      const targetFolders = subFolders.filter(f => f.isDirectory());

      if (targetFolders.length === 0) {
        console.log('하위 폴더가 없습니다');
        setModal(true);
        return;
      }

      const foldersWithDate = targetFolders
        .map(folder => {
          const date = parseDateFromFolderName(folder.name);
          return date ? { folder, date } : null;
        })
        .filter((item): item is { folder: typeof targetFolders[0]; date: Date } => item !== null);

      if (foldersWithDate.length === 0) {
        console.log('날짜 형식에 맞는 폴더가 없습니다');
        setModalTitle('데이터가 제대로 다운되지 않은 것 같아요.');
        setModalText('우측 상단 점 세 개-설정-개인 데이터 다운로드');
        setModal(true);
        return;
      }

      foldersWithDate.sort((a, b) => b.date.getTime() - a.date.getTime());
      const latestFolder = foldersWithDate[0];

      const now = new Date();
      const diff = now.getTime() - latestFolder.date.getTime();

      if (diff > TWO_WEEKS_MS) {
        setModalTitle('데이터가 오래되어 최신 데이터가 필요해요.');
        setModal(true);
        setModalText('우측 상단 점 세 개-설정-개인 데이터 다운로드');
        return;
      } else {
        setModal(false);
      }

      // const results = await biometricReader(latestFolder.folder.path); // 생체정보 실제로 읽어오기
      const results = testResults; // FOR TEST - 생체정보 테스트 파일로 읽어오기

      console.log('추출된 결과', results);
      // await saveResultsToFile(results);

      if (results && results.biometric_data_by_date.length > 0) {
        console.log('생체정보 가져오기 성공');
        navigation.replace('BiometricFetchComplete');
        dispatch(setBiometricInfo({ field: 'biometricData', value: results.biometric_data_by_date }));
      } else {
        setModalTitle('분석을 위해 연속 3일 데이터가 필요해요.');
        setModalText('연속 3일 이상 측정 후 다시 시도해주세요.');
        setModal(true);
        setIsThreeDay(false);

        const timer = setTimeout(() => {
          navigation.replace('Home');
        }, 3000);

        return () => clearTimeout(timer);
      }
    } catch (error) {
      console.log('읽기 에러:', error);
      setModalTitle('데이터가 제대로 다운되지 않은 것 같아요.');
      setModalText('우측 상단 점 세 개-설정-개인 데이터 다운로드');
      setModal(true);
    }
  };

  const isReading = useRef(false);

  useEffect(() => {
    const safeReadBiometricData = async () => {
      if (isReading.current) { return; }
      isReading.current = true;

      setTimeout(async () => {
      await readBiometricData();
      isReading.current = false;
    }, 1000);
    };

    safeReadBiometricData();

    const subscription = AppState.addEventListener('change', nextAppState => {
      if (
        appState.current.match(/inactive|background/) &&
        nextAppState === 'active'
      ) {
        safeReadBiometricData();
      }
      appState.current = nextAppState;
    });

    return () => {
      subscription.remove();
    };
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.textContainer}>
        <CustomText style={styles.title}>생체정보를 불러올게요!</CustomText>
      </View>

      <View style={styles.imageContainer}>
        <FastImage source={require('../../assets/images/biometric_start.gif')} style={styles.image} />
      </View>

      <Modal
        isVisible={modal}
        backdropColor="rgb(69, 69, 69)"
        backdropOpacity={0.3}
        animationIn="fadeIn"
        animationOut="fadeOut"
        style={styles.modalOverlay}
      >
        <View style={styles.modalContent}>
          <CustomText style={styles.modalTitle}>{modalTitle}</CustomText>
          <CustomText style={styles.modalText}>{modalText}</CustomText>

          {isThreeDay && <View style={styles.modalButtonWrapper}>
            <TouchableOpacity
              style={[styles.modalButton, styles.modalButtonSecondary]}
              onPress={() => {
                openSamsungHealth();
              }}
            >
              <CustomText style={styles.modalButtonText}>삼성헬스 열기</CustomText>
            </TouchableOpacity>
          </View>}
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  textContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  title: {
    fontSize: 24,
  },
  imageContainer: {
    flex: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 130,
    height: 130,
  },
  modalOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalContent: {
    alignItems: 'center',
    width: '90%',
    padding: 10,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
  },
  modalTitle: {
    marginVertical: 40,
    fontSize: 21,
  },
  modalText: {
    // marginVertical: 40,
    marginBottom: 40,
    fontSize: 17,
    color: '#575553',
  },
  modalButtonWrapper: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    borderRadius: 10,
  },
  modalButtonPrimary: {
    backgroundColor: '#F2EAE3',
  },
  modalButtonSecondary: {
    backgroundColor: '#D6CCC2',
  },
  modalButtonText: {
    fontSize: 18,
    color: '#575553',
  },
  modalContentText: {
    fontSize: 22,
    margin: 50,
  },
});

export default BiometricStartScreen;










// import React from 'react';
// import { View, Text, StyleSheet } from 'react-native';

// const BiometricStartScreen = () => {
//   // 전체 수면 시간 범위: 22:00 ~ 08:00 => 600분
//   const totalMinutes = 600;

//   const sleepStages = [
//     {
//       label: '얕은 수면',
//       start: '23:10',
//       end: '00:30',
//       color: '#F3F0EC',
//     },
//     {
//       label: '깊은 수면',
//       start: '00:30',
//       end: '02:20',
//       color: '#D8CCC2',
//     },
//     {
//       label: '렘 수면',
//       start: '02:20',
//       end: '04:50',
//       color: '#BFA392',
//     },
//     {
//       label: '각성 상태',
//       start: '04:50',
//       end: '06:40',
//       color: '#8E6F58',
//     },
//   ];

//   // 시간 문자열을 분 단위로 변환
//   // const timeToMinutes = (timeStr: string) => {
//   //   const [hour, minute] = timeStr.split(':').map(Number);
//   //   return hour * 60 + minute;
//   // };
//   const timeToMinutes = (time: string): number => {
//   const [h, m] = time.split(':').map(Number);
//   const minutes = h * 60 + m;
//   return h < 12 ? minutes + 1440 : minutes; // 오전이면 다음날로 간주
// };

//   // 기준 시작시간: 22:00 => 1320분
//   const baseStart = 22 * 60;

//   return (
//     <View style={styles.container}>
//       {/* 가로 막대 그래프 */}
//       <View style={styles.barContainer}>
//         {sleepStages.map((stage, index) => {
//           const startMin = timeToMinutes(stage.start);
//           const endMin = timeToMinutes(stage.end);
//           const offset = startMin - baseStart;
//           const duration = endMin - startMin;

//           console.log(`left: ${(offset / totalMinutes) * 100}%`);
//           console.log(`width: ${(duration / totalMinutes) * 100}%`);

//           return (
//             <View
//               key={index}
//               style={{
//                 position: 'absolute',
//                 left: `${(offset / totalMinutes) * 100}%`,
//                 width: `${(duration / totalMinutes) * 100}%`,
//                 // width: 50,
//                 height: 20,
//                 backgroundColor: stage.color,
//                 // backgroundColor: 'red',
//                 // borderRadius: 4,
//               }}
//             ><Text>{stage.label}</Text>
//               </View>
//           );
//         })}
//         {/* 배경 선 */}
//         <View style={styles.barBackground} />
//       </View>

//       {/* 시간 라벨 */}
//       <View style={styles.timeLabels}>
//         {['22시', '0시', '2시', '4시', '6시', '8시'].map((label, index) => (
//           <Text key={index} style={styles.timeLabel}>{label}</Text>
//         ))}
//       </View>

//       {/* 범례 */}
//       <View style={styles.legend}>
//         {sleepStages.map((stage, index) => (
//           <View key={index} style={styles.legendRow}>
//             <View style={[styles.colorDot, { backgroundColor: stage.color }]} />
//             <Text style={styles.legendText}>
//               {stage.label} {stage.start}~{stage.end}
//             </Text>
//           </View>
//         ))}
//       </View>
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: { padding: 16 },
//   barContainer: {
//     height: 20,
//     backgroundColor: '#eee',
//     borderRadius: 4,
//     marginBottom: 8,
//     position: 'relative',
//     overflow: 'hidden',
//   },
//   barBackground: {
//     ...StyleSheet.absoluteFillObject,
//     borderWidth: 1,
//     borderColor: '#ccc',
//     borderRadius: 4,
//   },
//   timeLabels: {
//     flexDirection: 'row',
//     justifyContent: 'space-between',
//     marginTop: 4,
//   },
//   timeLabel: {
//     fontSize: 12,
//     color: '#888',
//   },
//   legend: {
//     marginTop: 12,
//   },
//   legendRow: {
//     flexDirection: 'row',
//     alignItems: 'center',
//     marginBottom: 4,
//   },
//   colorDot: {
//     width: 12,
//     height: 12,
//     borderRadius: 6,
//     marginRight: 8,
//   },
//   legendText: {
//     fontSize: 13,
//     color: '#444',
//   },
// });

// export default BiometricStartScreen;
