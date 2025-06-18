import React, { useEffect, useState } from 'react';
import { ScrollView, View, TouchableOpacity, ActivityIndicator, StyleSheet, Image, Linking } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { AppDispatch, RootState } from '../../redux/store.ts';
import { loadTokens, logAllAsyncStorage } from '../../redux/actions/authAction.ts';
import { setUserInfo, fetchUser } from '../../redux/slices/userSlice.ts';
import { setBiometricInfo } from '../../redux/slices/biometricSlice.ts';
import { fetchLifestyle } from '../../redux/slices/lifestyleSlice.ts';
import { fetchReport } from '../../redux/slices/reportSlice.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';
import { DEMENTIA_INFO } from '../../constants/dementiaInfo.ts';

const HomeScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const name = useSelector((state: RootState) => state.user.userInfo.name);
  const lifestyleInfo = useSelector((state: RootState) => state.lifestyle);
  const biometricInfo = useSelector((state: RootState) => state.biometric);

  // const fetchResult = useSelector((state: RootState) => state.lifestyle.fetchResult);
  // console.log('fetchResult:', fetchResult);

  const [isLifestyleSaved, setIsLifestyleSaved] = useState<boolean>(false);
  const [isBiometricSaved, setIsBiometricSaved] = useState<boolean>(false);
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [randomItems, setRandomItems] = useState([]);

  // console.log('비어있나 확인:', userInfo);

  // const genderValue = userInfo.gender === '남' ? '0' : '1';
  // const year = userInfo.birthdate.slice(0, 4);
  // const month = userInfo.birthdate.slice(5, 7);
  // const day = userInfo.birthdate.slice(8, 10);

  // console.log(genderValue, year, month, day);

  // useEffect(() => {
  //   const handleFetch = async () => {
  //     const { accessToken } = await loadTokens();
  //     if (!accessToken) {
  //       console.log('로그인 정보가 없습니다.');
  //       return;
  //     }

  //     dispatch(fetchUser(accessToken));
  //   };
  //   handleFetch();
  // }, []);

  useEffect(() => {
    // 치매 알아보기
    const shuffleArray = (array: any) => {
      const copy = [...array];
      for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
      }
      return copy;
    };
    const shuffled = shuffleArray(DEMENTIA_INFO);
    setRandomItems(shuffled.slice(0, 3));
  }, []);

  // 라이프스타일 입력 완료 여부 확인
  useEffect(() => {
    const handleFetchLifestyle = async () => {
      const { accessToken } = await loadTokens();
      // console.log('accessToken:', accessToken);
      if (!accessToken) {
        console.log('로그인 정보가 없습니다.');
        return;
      }

      // 2
      // try {
      //   const decoded: { exp: number } = jwtDecode(accessToken);
      //   const currentTime = Date.now() / 1000;

      //   if (decoded.exp < currentTime) {
      //     console.log('토큰이 만료되었습니다.');
      //     return;
      //   }

      //   dispatch(fetchLifestyle(accessToken));
      // } catch (error) {
      //   console.log('토큰 디코딩 실패:', error);
      // }

      // 3
      // const loggedInUser = await AsyncStorage.getItem('loggedInUser');

      // if (!loggedInUser) {
      //   console.log('저장된 사용자 정보가 없습니다.');
      //   return;
      // }

      // const parsedUser = JSON.parse(loggedInUser);

      // if (parsedUser.email !== '') {
      //   const { accessToken } = await loadTokens();
      //   if (!accessToken) {
      //     console.log('로그인 정보가 없습니다.');
      //     return;
      //   }
      //   dispatch(fetchLifestyle(accessToken));
      // }

      dispatch(fetchLifestyle(accessToken));
    };

    handleFetchLifestyle();
  }, []);

  // 라이프스타일 입력 완료 여부 반영
  useEffect(() => {
    if (lifestyleInfo.fetchResult?.message === '라이프스타일 조회 성공') {
      setIsLifestyleSaved(true);
    } else {
      setIsLifestyleSaved(false);
    }
  }, [lifestyleInfo.fetchResult]);

  // 생체정보 입력 완료 여부 확인, 반영
  useEffect(() => {
    const handleFetchBiometric = async () => {
      const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');

      if (biometricInfoInStorage) {
        const biometricInfoStr = JSON.parse(biometricInfoInStorage);
        if (userInfo.email === biometricInfoStr?.email) {
          dispatch(setBiometricInfo({ field: 'biometricData', value: biometricInfoStr.biometricInfo}));
          // dispatch(setBiometricInfo({ field: 'biometricData', value: biometricInfoStr.biometricInfo.biometricData}));
          setIsBiometricSaved(true);
        } else {
          setIsBiometricSaved(false);
        }
      }
    };

    handleFetchBiometric();
  }, []);

  useEffect(() => {
    console.log('생체정보 그대로 있나 확인:', biometricInfo);
  }, []);

  const [asyncLoading, setAsyncLoading] = useState(true);

  useEffect(() => {
    const handleFetchBiometric = async () => {
      try {
        const biometricInfoInStorage = await AsyncStorage.getItem('biometricInfo');

        if (biometricInfoInStorage) {
          const biometricInfoStr = JSON.parse(biometricInfoInStorage);
          if (userInfo.email === biometricInfoStr?.email) {
            dispatch(setBiometricInfo({ field: 'biometricData', value: biometricInfoStr.biometricInfo }));
            setIsBiometricSaved(true);
          } else {
            setIsBiometricSaved(false);
          }
        }
      } catch (error) {
        console.error('Error fetching biometric info:', error);
      } finally {
        setAsyncLoading(false); // 이 시점에 로딩 끝남
      }
    };

    handleFetchBiometric();
  }, []);

  useEffect(() => {
    logAllAsyncStorage();
  }, []);

  // 진단 결과 불러오기
  useEffect(() => {
    const loadReports = async () => {
      try {
        const { accessToken } = await loadTokens();
        // console.log('accessToken:', accessToken);
        if (!accessToken) {
          console.log('로그인 정보가 없습니다.');
          return;
        }
        // const token = await AsyncStorage.getItem('accessToken');
        const createDateJson = await AsyncStorage.getItem('createDate');
        const dateList: number[] = createDateJson ? JSON.parse(createDateJson) : [];

        const reportResults = await Promise.all(
          dateList.map(async (date) => {
            const result = await dispatch(fetchReport({ token: accessToken, date }));
            if (fetchReport.fulfilled.match(result)) {
              return { date, data: result.payload };
            } else {
              return null;
            }
          })
        );

        const validReports = reportResults.filter((item) => item !== null);
        setReports(validReports);
      } catch (e) {
        console.log('진단 결과 불러오기 실패', e);
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, [dispatch]);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <TouchableOpacity
          style={styles.loginContainer}
          onPress={() => {
            // typeof name === 'string' ? navigation.navigate('Mypage') : navigation.navigate('Login');
            name !== '' ? navigation.navigate('Mypage') : navigation.navigate('Login');
          }}
        >
        {/* {typeof name === 'string' ? ( */}
        {name !== '' ? (
          <CustomText style={styles.loginText}>{`${name}님, 안녕하세요!`}</CustomText>
        ) : (
          <CustomText style={styles.loginText}>로그인이 필요합니다.</CustomText>
        )}

          <Icon name="chevron-forward" size={16} color="gray" />
        </TouchableOpacity>

        <View style={styles.section}>
          <CustomText weight="bold" style={styles.sectionTitle}>치매 진단하기</CustomText>
          <View style={styles.card}>
            <TouchableOpacity style={styles.row} onPress={() => {isLifestyleSaved ? navigation.navigate('LifestyleView') : navigation.navigate('LifestyleStart');}}>
              <CustomText style={styles.cardText}>라이프스타일 입력</CustomText>
              <View style={styles.statusContainer}>
                <CustomText style={[styles.status, isLifestyleSaved && { color: '#434240' }]}>{isLifestyleSaved ? '완료' : '미완'}</CustomText>
                <Icon name="chevron-forward" size={16} color="gray" />
              </View>
            </TouchableOpacity>
            {/* <TouchableOpacity style={styles.row} onPress={() => {isBiometricSaved ? navigation.navigate('BiometricOverview', { from: 'Home' }) : navigation.navigate('BiometricStart')}}> */}
              <TouchableOpacity
                style={styles.row}
                onPress={() => {
                  if (asyncLoading) return;
                  isBiometricSaved
                    ? navigation.navigate('BiometricOverview', { from: 'Home' })
                    : navigation.navigate('BiometricStart');
                }}
                disabled={asyncLoading}
              >
              <CustomText style={styles.cardText}>생체정보 입력</CustomText>
              <View style={styles.statusContainer}>
                {asyncLoading ?
                <ActivityIndicator size="small" color="#888" />
                :
                <CustomText style={[styles.status, isBiometricSaved && { color: '#434240' }]}>{isBiometricSaved ? '완료' : '미완'}</CustomText>
                }
                {/* <CustomText style={[styles.status, isBiometricSaved && { color: '#434240' }]}>{isBiometricSaved ? '완료' : '미완'}</CustomText> */}
                <Icon name="chevron-forward" size={16} color="gray" />
              </View>
            </TouchableOpacity>
          </View>
          <TouchableOpacity style={[styles.button, isLifestyleSaved && isBiometricSaved && { backgroundColor: '#917A6B' }]} onPress={() => navigation.navigate('ReportStart')} disabled={!(isLifestyleSaved && isBiometricSaved)}>
            <CustomText style={[styles.buttonText, isLifestyleSaved && isBiometricSaved && { color: '#FFFFFF' }]}>진단하기</CustomText>
          </TouchableOpacity>
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>진단 결과 보기</CustomText>
            <TouchableOpacity onPress={() => navigation.navigate('ReportView', { from : 'Home' })}>
              <CustomText style={styles.moreText}>더보기</CustomText>
            </TouchableOpacity>
          </View>
          {loading ? (
            <View style={styles.reportHistoryBox}>
              <ActivityIndicator size="small" color="#888" />
            </View>
          ) : reports.length > 0 ? (
            <View style={styles.reportHistoryBox}>
              {reports
                .sort((a, b) => b.date - a.date)
                .slice(0, 4)
                .map((report, index) => (
                  <TouchableOpacity
                    key={index}
                    style={[styles.reportHistoryRow, (index <= 3 && index === reports.length - 1) && { marginBottom: 0 }]}
                    // style={styles.reportHistoryRow}
                    onPress={() =>
                      navigation.navigate('ReportResult', {
                        type: 'Home',
                        data: report.data,
                      })
                    }
                  >
                    <CustomText style={styles.reportHistoryText}>
                      {`${report.date.toString().slice(2, 4)}/${report.date.toString().slice(4, 6)}/${report.date.toString().slice(6, 8)}`}
                    </CustomText>
                    <Icon name="chevron-forward" size={16} color="gray" />
                  </TouchableOpacity>
                ))}
                {/* <TouchableOpacity style={[styles.reportHistoryRow]}>
                  <CustomText style={styles.reportHistoryText}>25/04/12</CustomText>
                  <Icon name="chevron-forward" size={16} color="gray" />
                </TouchableOpacity>
                <TouchableOpacity style={[styles.reportHistoryRow]}>
                  <CustomText style={styles.reportHistoryText}>25/01/26</CustomText>
                  <Icon name="chevron-forward" size={16} color="gray" />
                </TouchableOpacity>
                <TouchableOpacity style={[styles.reportHistoryRow, {marginBottom: 0}]}>
                  <CustomText style={styles.reportHistoryText}>24/10/30</CustomText>
                  <Icon name="chevron-forward" size={16} color="gray" />
                </TouchableOpacity> */}
            </View>
          ) : (
            <View style={styles.resultBox}>
              <CustomText style={styles.resultText}>진단 결과가 없습니다.</CustomText>
            </View>
          )}
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>치매 알아보기</CustomText>
            <TouchableOpacity onPress={() => navigation.navigate('DementiaInfoList')}>
              <CustomText style={styles.moreText}>더보기</CustomText>
            </TouchableOpacity>
          </View>
          {randomItems.map((item) => (
            <TouchableOpacity key={item.id} style={styles.infoCard} onPress={() => navigation.navigate('DementiaInfoDetail', { item })}>
              <Image source={item.image} style={{ width: 20, height: 20, marginRight: 10 }} />
              <CustomText style={styles.infoText}>{item.title}</CustomText>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.sectionSecondary}>
          <View style={styles.row}>
            <CustomText weight="bold" style={styles.sectionTitle}>치매 예방하기</CustomText>
          </View>
            <TouchableOpacity style={styles.preventionCard} onPress={() => Linking.openURL('https://www.youtube.com/watch?v=6jJ3sauw7mQ&t=763s')}>
              <View>
                <CustomText style={styles.preventionTitle}>치매 예방 체조</CustomText>
                <CustomText style={styles.preventionContent}>보건복지부와 중앙치매센터가 전문가들과 개발했어요.</CustomText>
              </View>
              <View>
                <Image source={require('../../assets/images/yoga-pose.png')} style={styles.image} />
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.preventionCard} onPress={() => Linking.openURL('https://www.youtube.com/watch?v=lIjliKXCaSY')}>
              <View>
                <CustomText style={styles.preventionTitle}>치매 예방 박수</CustomText>
                <CustomText style={styles.preventionContent}>손을 움직여 뇌를 자극해요.</CustomText>
              </View>
              <View>
                <Image source={require('../../assets/images/clap.png')} style={styles.image} />
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.preventionCard} onPress={() => Linking.openURL('https://www.youtube.com/watch?v=Q70mbW9Az4k&t=10s')}>
              <View>
                <CustomText style={styles.preventionTitle}>치매 테스트</CustomText>
                <CustomText style={styles.preventionContent}>뇌신경센터 기억력 테스트로 치매를 예방해요.</CustomText>
              </View>
              <View>
                <Image source={require('../../assets/images/memory.png')} style={styles.image} />
              </View>
            </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  scrollContent: {
    paddingVertical: 10,
    paddingHorizontal: 16,
  },
  loginContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
  },
  loginText: {
    fontSize: 24,
  },
  section: {
    marginTop: 20,
  },
  sectionSecondary: {
    marginTop: 30,
  },
  sectionTitle: {
    fontSize: 24,
    marginBottom: 8,
  },
  card: {
    marginTop: 8,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  cardText: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  status: {
    fontSize: 16,
    color: '#9F9E9B',
  },
  button: {
    alignItems: 'center',
    padding: 10,
    marginTop: 10,
    borderRadius: 10,
    backgroundColor: '#F2EFED',
  },
  buttonText: {
    fontSize: 18,
    color: '#B4B4B4',
  },
  moreText: {
    fontSize: 16,
    color: '#868481',
  },
  resultBox: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 50,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  resultText: {
    fontSize: 18,
    color: '#6C6C6B',
  },
  reportHistoryBox: {
    // alignItems: 'flex-start',
    // paddingVertical: 12,
    paddingTop: 16,
    // paddingBottom: 6,
    paddingBottom: 16,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  reportHistoryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  reportHistoryText: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  infoCard: {
    flexDirection: 'row',
    padding: 12,
    marginVertical: 5,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  infoText: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  preventionCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingLeft: 12,
    paddingVertical: 20,
    marginVertical: 5,
    borderRadius: 10,
    backgroundColor: '#F2EAE3',
  },
  preventionTitle: {
    paddingLeft: 6,
    fontSize: 18,
    color: '#6C6C6B',
  },
  preventionContent: {
    marginTop: 4,
    paddingLeft: 6,
    fontSize: 14,
    color: '#6C6C6B',
  },
  image: {
    width: 50,
    height: 50,
    marginRight: 20,
  },
});

export default HomeScreen;