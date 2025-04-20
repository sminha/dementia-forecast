import React from 'react';
import { View, SafeAreaView, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useDispatch } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import { setFormData } from '../../redux/slices/signupSlice.ts';
import { AppDispatch } from '../../redux/store.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const PrivacySharePolicySreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
      </SafeAreaView>

      <View style={styles.headerContainer}>
        <CustomText style={styles.titleText}>개인정보 제공 동의</CustomText>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContentContainer} overScrollMode="never">
        <CustomText style={styles.sectionTitle}>개인정보 제공 동의</CustomText>
        <CustomText style={styles.contentText}>
        치매예보(이하 “회사”)가 제공하는 본 서비스를 이용함에 있어, 「개인정보 보호법」 제17조 및 제18조에 따라 개인정보를 제3자에게 제공하기 위하여 정보주체인 귀하의 동의가 필요합니다. 회사는 사용자의 건강과 삶의 질을 증진시키기 위해 수집한 정보를 안전하게 관리하고, 아래와 같은 목적에 한하여 필요한 범위 내에서 개인정보를 제3자에게 제공하고자 합니다.{'\n'}
        회사는 치매 고위험군으로 판단되는 경우, 사용자의 건강과 안전을 위해 치매안심센터, 행정복지센터 관련 부서 등 관련 공공기관에 정보를 전달하여 사용자가 적절한 조치를 지원받을 수 있도록 합니다. 그 외의 경우에는 개인정보를 제3자에게 제공하지 않으며, 외부에 업무를 위탁하는 경우에도 사용자를 식별할 수 없도록 안전하게 처리하고, 개인정보 유출 방지를 위해 최선을 다하겠습니다.{'\n'}
        ※ 귀하는 위 개인정보 제공에 대해 동의를 거부할 권리가 있습니다. 다만, 해당 정보는 위험군 사용자에 대한 공공기관 연계 조치를 위한 필수 정보이므로 동의를 거부하실 경우 해당 지원이 제한될 수 있습니다.
        </CustomText>

        <CustomText style={styles.sectionTitle}>제공 목적</CustomText>
        <CustomText style={styles.contentText}>
          치매 고위험 사용자에 대한 조기 대응 및 적절한 공공 지원 연계
        </CustomText>

        <CustomText style={styles.sectionTitle}>제공 대상</CustomText>
        <CustomText style={styles.contentText}>
          치매안심센터, 행정복지센터 등 사용자 주변의 관련 공공기관
        </CustomText>

        <CustomText style={styles.sectionTitle}>제공 항목</CustomText>
        <CustomText style={styles.contentText}>
          가. 개인정보: 이름, 성별, 생년월일, 성별, 전화번호, 주소{'\n'}
          나. 라이프스타일 정보: 가구원 수, 배우자 유무, 월 소득, 월 지출, 월 음주 지출, 월 담배 지출, 월 서적 지출, 월 의료 지출, 월 보험비, 애완동물 유무, 가정 형태, 가구돌봄유형{'\n'}
          다. 생체정보: 걸음 수, 활동 시간, 활동 칼로리, 수면 시간, 수면 단계{'\n'}
          라. 예측정보: 치매 위험도, 분석 레포트
        </CustomText>
      </ScrollView>

      <View>
        <TouchableOpacity onPress={() => {
          dispatch(setFormData({field: 'privacyShare', value: true}));
          navigation.goBack();
          }}
          style={styles.actionButton}>
          <CustomText style={styles.actionButtonText}>동의합니다.</CustomText>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
  },
  safeAreaWrapper: {
    width: 70,
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  headerContainer: {
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  backButton: {
    marginBottom: 10,
  },
  titleText: {
    fontSize: 24,
  },
  scrollContentContainer: {
    paddingHorizontal: 20,
  },
  chevronIcon: {
    marginLeft: 'auto',
  },
  sectionTitle: {
    marginBottom: 20,
    fontSize: 20,
  },
  contentText: {
    marginBottom: 20,
    fontSize: 16,
    lineHeight: 24,
    color: '#202020',
  },
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    borderRadius: 5,
    backgroundColor: '#D6CCC2',
  },
  actionButtonText: {
    fontSize: 20,
    color: '#575553',
  },
});

export default PrivacySharePolicySreen;