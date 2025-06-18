import React from 'react';
import { View, SafeAreaView, TouchableOpacity, Image, ScrollView, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../types/navigationTypes.ts';
import CustomText from '../../components/CustomText.tsx';
import Icon from '../../components/Icon.tsx';
import { DEMENTIA_INFO } from '../../constants/dementiaInfo.ts';

const DementiaInfoListScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'EmailLogin'>;
  const navigation = useNavigation<Navigation>();;

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeAreaWrapper}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Icon name="chevron-back" size={16} />
        </TouchableOpacity>
        <CustomText style={styles.title}>치매 알아보기</CustomText>
      </SafeAreaView>

      <ScrollView showsVerticalScrollIndicator={false} overScrollMode="never">
        {DEMENTIA_INFO.map((item) => (
          <TouchableOpacity key={item.id} style={styles.infoCard} onPress={() => {navigation.navigate('DementiaInfoDetail', { item })}}>
            <Image source={item.image} style={{ width: 20, height: 20, marginRight: 10 }} />
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
    width: 60,
    // left: 10,
    paddingLeft: 10,
    paddingVertical: 23,
    // marginLeft: 10,
  },
  title: {
    fontSize: 24,
    textAlign: 'center',
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
});

export default DementiaInfoListScreen;