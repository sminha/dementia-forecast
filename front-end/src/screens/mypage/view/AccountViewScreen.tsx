import React, { useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { AppDispatch, RootState } from '../../../redux/store.ts';
import { loadTokens } from '../../../redux/actions/authAction.ts';
import { fetchUser } from '../../../redux/slices/userSlice.ts';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const AccountViewScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);

  function formatISOToKoreanDate(isoString: string): string {
    const date = new Date(isoString);

    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');

    return `${year}년 ${month}월 ${day}일`;
  }

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
  // }, [dispatch]);

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
          <View style={styles.titleWrapper}>
            <CustomText style={styles.loginText}>계정 정보</CustomText>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.list}>
            <View style={styles.listRow}>
              <CustomText style={styles.listTitle}>이메일</CustomText>
              <View style={styles.listContentPrimary}>
                <CustomText style={styles.listContentText}>{userInfo.email}</CustomText>
              </View>
            </View>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('PasswordEdit')}>
              <CustomText style={styles.listTitle}>비밀번호</CustomText>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('NameEdit')}>
              <CustomText style={styles.listTitle}>이름</CustomText>
              <View style={styles.listContent}>
                <CustomText style={styles.listContentText}>{userInfo.name}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('GenderEdit')}>
              <CustomText style={styles.listTitle}>성별</CustomText>
              <View style={styles.listContent}>
                <CustomText style={styles.listContentText}>{userInfo.gender}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('BirthdateEdit')}>
              <CustomText style={styles.listTitle}>생년월일</CustomText>
              <View style={styles.listContent}>
                <CustomText style={styles.listContentText}>{formatISOToKoreanDate(userInfo.birthdate)}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('PhoneEdit')}>
              <CustomText style={styles.listTitle}>전화번호</CustomText>
              <View style={styles.listContent}>
                <CustomText style={styles.listContentText}>{userInfo.phone.replace(/(\d{3})(\d{4})(\d{4})/, '$1-$2-$3')}</CustomText>
              </View>
              <Icon name="chevron-forward" size={16} color="gray" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.listRow} onPress={() => navigation.navigate('AddressEdit')}>
              <CustomText style={styles.listTitle}>주소</CustomText>
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
  listTitle: {
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
    position: 'relative',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
  },
    listContentPrimary: {
    position: 'absolute',
    left: 0,
    right: 5,
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginBottom: 3,
  },
  listContent: {
    position: 'absolute',
    left: 0,
    right: 30,
    alignItems: 'flex-end',
    justifyContent: 'center',
    marginBottom: 3,
  },
  listContentText: {
    fontSize: 18,
    color: '#434240',
  },
});

export default AccountViewScreen;