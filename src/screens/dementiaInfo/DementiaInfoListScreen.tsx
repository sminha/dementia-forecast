import React from 'react';
import { View, SafeAreaView, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';

const DementiaInfoListScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();;

    const dementiaInfo = [
    { id: '1', title: '🧬  치매는 유전병일까요? ', content: '치매는 단일 질병이 아닌, 다양한 원인에 의해 발생하는 증후군입니다. 가장 흔한 형태는 알츠하이머병이며, 그 외에도 혈관성 치매, 루이소체 치매 등이 있습니다. 많은 분들이 치매가 유전되는 병인지 궁금해하시지만, 결론부터 말씀드리자면 치매는 일부 유전적 요인을 가질 수 있지만, 대부분의 경우 유전병은 아닙니다.\n\n알츠하이머병의 경우, 드물게 40~50대에 발병하는 조기 발병형은 특정 유전자 변이로 인해 가족력을 보이는 경우가 있습니다. 하지만 전체 알츠하이머병의 1% 미만으로 매우 드문 경우입니다. 일반적인 노년기 치매는 유전보다는 나이, 생활습관, 심혈관 건강 등의 환경적 요인이 더 크게 작용합니다.\n\n즉, 가족 중에 치매 환자가 있다고 해서 반드시 치매에 걸리는 것은 아니며, 건강한 식습관, 운동, 두뇌 활동을 유지하는 것만으로도 발병 위험을 낮출 수 있습니다. 유전이라는 단어에 너무 두려움을 갖기보다는, 지금부터의 관리가 훨씬 더 중요하다는 점을 기억하시면 좋겠습니다.' },
    { id: '2', title: '👵🏻  치매는 노인들만 걸리는 병일까요? ', content: '' },
    { id: '3', title: '🤒  치매와 알츠하이머는 같은 병일까요? ', content: '' },
    { id: '4', title: '💭  치매와 단순 건망증은 어떻게 구별하나요? ', content: '' },
    { id: '5', title: '😄  치매는 완치가 가능한가요? ', content: '' },
    { id: '6', title: '☝🏻  치매 초기 증상은 어떤 게 있나요? ', content: '' },
    { id: '7', title: '✍🏻  치매 자가 진단은 어떻게 하나요? ', content: '' },
    { id: '8', title: '😶‍🌫️  우울증이 치매를 유발할 수 있나요? ', content: '' },
    { id: '9', title: '💊  약물로 치매 진행을 늦출 수 있을까요? ', content: '' },
    { id: '10', title: '👥  치매 환자 가족은 어떻게 대처하면 좋을까요? ', content: '' },
    { id: '11', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '12', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '13', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '14', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '15', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '16', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
    { id: '17', title: '💬  언어 능력 저하도 치매 증상인가요? ', content: '' },
  ];

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
        <CustomText style={styles.title}>치매 알아보기</CustomText>
      </SafeAreaView>

      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        {dementiaInfo.map((item) => (
          <TouchableOpacity key={item.id} style={styles.infoCard} onPress={() => {navigation.navigate('DementiaInfoDetail')}}>
            <CustomText style={styles.infoText}>{item.title}</CustomText>
          </TouchableOpacity>
        ))}
      </ScrollView>
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
    // without title
    // paddingVertical: 16,
    // paddingHorizontal: 10,

    // with title
    position: 'relative',
    paddingVertical: 16,
    paddingHorizontal: 10,
  },
  backButton: {
    // without title
    // width: 20,
    // paddingTop: 7,

    // with title
    position: 'absolute',
    width: 40,
    left: 10,
    paddingVertical: 23,
  },
  title: {
    fontSize: 24,
    textAlign: 'center',
  },
  infoCard: {
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
});

export default DementiaInfoListScreen;