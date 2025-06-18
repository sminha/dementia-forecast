import React, { useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import { useSelector, useDispatch } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { RootState, AppDispatch } from '../../../redux/store.ts';
import { loadTokens } from '../../../redux/actions/authAction.ts';
import { fetchLifestyle } from '../../../redux/slices/lifestyleSlice.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const LifestyleViewScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const lifestyleInfo = useSelector((state: RootState) => state.lifestyle.lifestyleInfo);

  useEffect(() => {
    const handleFetch = async () => {
      const { accessToken } = await loadTokens();
      if (!accessToken) {
        console.log('로그인 정보가 없습니다.');
        return;
      }

      dispatch(fetchLifestyle(accessToken));
    };

    handleFetch();
  }, [dispatch]);

  // console.log(lifestyleInfo);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
          <View style={styles.titleWrapper}>
            <CustomText style={styles.loginText}>라이프스타일</CustomText>
          </View>
        </View>


        <View style={styles.section}>
          <View style={styles.list}>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '가구원 수' })}>
              <CustomText style={styles.listText}>가구원 수</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.householdSize}명</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '배우자 유무' })}>
              <CustomText style={styles.listText}>배우자 유무</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.hasSpouse ? '있음' : '없음'}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 소득' })}>
              <CustomText style={styles.listText}>월 소득</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.income}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 지출' })}>
              <CustomText style={styles.listText}>월 지출</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.expenses}만원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 음주 지출' })}>
              <CustomText style={styles.listText}>월 음주 지출</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.alcoholExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 담배 지출' })}>
              <CustomText style={styles.listText}>월 담배 지출</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.tobaccoExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 서적 지출' })}>
              <CustomText style={styles.listText}>월 서적 지출</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.bookExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 복지시설 지출' })}>
              <CustomText style={styles.listText}>월 복지시설 지출</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.welfareExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 의료비' })}>
              <CustomText style={styles.listText}>월 의료비</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.medicalExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '월 보험비' })}>
              <CustomText style={styles.listText}>월 보험비</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.insuranceExpense}만 원</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '애완동물 유무' })}>
              <CustomText style={styles.listText}>애완동물 유무</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.hasPet ? '있음' : '없음'}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('LifestyleEdit', { topic: '가정 형태' })}>
              <CustomText style={styles.listText}>가정 형태</CustomText>
              <View style={styles.result}>
                <CustomText style={styles.resultText}>{lifestyleInfo.householdType}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
          </View>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
    height: 50,
  },
  backContainer: {
    paddingHorizontal: 4,
    zIndex: 1,
  },
  titleWrapper: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
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
  list: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  listText: {
    fontSize: 18,
    color: '#434240',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingRight: 5,
  },
  listRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
    position: 'relative',
  },
  result: {
    position: 'absolute',
    left: 0,
    right: 30,
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginBottom: 3,
  },
  resultText: {
    fontSize: 18,
    color: '#434240',
  }
});

export default LifestyleViewScreen;