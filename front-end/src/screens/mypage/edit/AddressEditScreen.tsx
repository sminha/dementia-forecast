import React, { useState, useEffect } from 'react';
import { ScrollView, View, TouchableOpacity, TextInput, StyleSheet } from 'react-native';
import { useDispatch, useSelector } from 'react-redux';
import Modal from 'react-native-modal';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../../types/navigationTypes.ts';
import { setUserInfo, updateUser, clearUpdateResult } from '../../../redux/slices/userSlice.ts';
import { AppDispatch, RootState } from '../../../redux/store.ts';
import { loadTokens } from '../../../redux/actions/authAction.ts';
import Postcode from '@actbase/react-daum-postcode';
import CustomText from '../../../components/CustomText.tsx';
import Icon from 'react-native-vector-icons/Ionicons';

const AddressEditScreen = () => {
  type Navigation = StackNavigationProp<RootStackParamList, 'Home'>;
  const navigation = useNavigation<Navigation>();

  const dispatch = useDispatch<AppDispatch>();
  const userInfo = useSelector((state: RootState) => state.user.userInfo);
  const updateResult = useSelector((state: RootState) => state.user.updateResult);

  const [address, setAddress] = useState('');
  const [detailAddress, setDetailAddress] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [isPostcodeVisible, setPostcodeVisible] = useState(false);

  useEffect(() => {
    // if (updateResult?.statusCode === 200) {
    if (updateResult?.message === '회원 정보가 수정되었습니다.') {
      dispatch(setUserInfo({ field: 'address', value: `${address} ${detailAddress}` }));
      dispatch(clearUpdateResult());
      navigation.goBack();
    // } else if (updateResult && updateResult.statusCode !== 200) {
    } else if (updateResult && updateResult.message !== '회원 정보가 수정되었습니다.') {
      dispatch(clearUpdateResult());
    }
  }, [updateResult, address, detailAddress, navigation, dispatch]);

  const handleFocus = (field: string) => {
    setFocusedField(field);
  };

  const handleBlur = () => {
    setFocusedField(null);
  };

  const isFormValid = () => {
    return address && detailAddress && (userInfo.address !== `${address} ${detailAddress}`);
  };

  const handleUpdate = async () => {
    const { accessToken } = await loadTokens();
    if (!accessToken) {
      console.log('로그인 정보가 없습니다.');
      return;
    }

    dispatch(updateUser({ token: accessToken, userInfo: { ...userInfo, address: `${address} ${detailAddress}` } }));
    // dispatch(updateUser({ token: accessToken, userInfo }));
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} overScrollMode="never">
        <View style={styles.header}>
          <TouchableOpacity style={styles.backContainer} onPress={() => navigation.goBack()}>
            <Icon name="chevron-back" size={16} color="gray" />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <View style={styles.title}>
            <TouchableOpacity style={styles.titleRow}>
              <CustomText style={styles.titleText}>주소 수정하기</CustomText>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.row}>
          <CustomText style={[styles.label, focusedField === 'detailAddress' && styles.focusedLabel]}>주소</CustomText>
          <TouchableOpacity
            onPress={() => setPostcodeVisible(true)}
            style={[styles.addressInput, focusedField === 'detailAddress' && styles.focusedInput]}
          >
            <CustomText style={styles.addressText}>{address}</CustomText>
          </TouchableOpacity>
          <TextInput
            style={[styles.detailAddressInput, focusedField === 'detailAddress' && styles.focusedInput]}
            onFocus={() => handleFocus('detailAddress')}
            onBlur={() => handleBlur()}
            onChangeText={(text) => {
              setDetailAddress(text);
              // dispatch(setUserInfo({ field: 'detailAddress', value: text }));
            }}
          />
        </View>

        <View>
          <TouchableOpacity onPress={handleUpdate} style={[styles.actionButton, isFormValid() ? styles.actionButtonEnabled : styles.actionButtonDisabled]} disabled={!isFormValid()}>
            <CustomText style={[styles.actionButtonText, isFormValid() ? styles.actionButtonTextEnabled : styles.actionButtonTextDisabled]}>수정 완료</CustomText>
          </TouchableOpacity>
        </View>

        <Modal
          isVisible={isPostcodeVisible}
          onBackdropPress={() => setPostcodeVisible(false)}
          onBackButtonPress={() => setPostcodeVisible(false)}
          backdropColor="rgb(69, 69, 69)"
          backdropOpacity={0.3}
          animationIn="fadeIn"
          animationOut="fadeOut"
        >
          <View style={styles.postcodeContainer}>
            <Postcode
              style={styles.postcode}
              jsOptions={{ animation: true }}
              onSelected={(data) => {
                setAddress(data.address);
                // dispatch(setUserInfo({ field: 'address', value: data.address }));
                setPostcodeVisible(false);
              }}
              onError={() => setPostcodeVisible(false)}
            />
          </View>
        </Modal>
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
  section: {
    marginTop: 20,
  },
  title: {
    paddingHorizontal: 5,
    paddingTop: 10,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    // marginBottom: 20,
    marginBottom: 30,
  },
  titleText: {
    fontSize: 24,
  },
  inputGroup: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  labelText: {
    fontSize: 18,
    color: '#202020',
  },
  labelTextFocused: {
    color: '#9F8473',
  },
  inputFieldWrapper: {
    position: 'relative',
    flexDirection: 'row',
    alignItems: 'center',
  },
  clearButton: {
    position: 'absolute',
    top: '50%',
    right: 1,
    transform: [{ translateY: -8 }],
  },
  inputField: {
    width: '100%',
    paddingRight: 35,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  inputFieldFocused: {
    borderBottomColor: '#9F8473',
  },
  actionButton: {
    alignItems: 'center',
    padding: 15,
    marginTop: 10,
    marginBottom: 5,
    marginHorizontal: 5,
    borderRadius: 5,
  },
  actionButtonEnabled: {
    backgroundColor: '#D6CCC2',
  },
  actionButtonDisabled: {
    backgroundColor: '#F2EFED',
  },
  actionButtonText: {
    fontSize: 20,
  },
  actionButtonTextEnabled: {
    color: '#575553',
  },
  actionButtonTextDisabled: {
    color: '#B4B4B4',
  },




  row: {
    marginBottom: 40,
    marginHorizontal: 5,
  },
  rowWithoutMatch: {
    marginBottom: 60,
  },
  label: {
    fontSize: 18,
    color: '#202020',
  },
  focusedLabel: {
    color: '#9F8473',
  },
  addressInput: {
    height: 45,
    marginTop: 5,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    justifyContent: 'center',
  },
  addressText: {
    marginLeft: 4,
    fontSize: 18,
  },
  detailAddressInput: {
    marginTop: 20,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#B4B4B4',
    fontSize: 18,
  },
  focusedInput: {
    borderBottomColor: '#9F8473',
  },
  postcodeContainer: {
    flex: 0.7,
  },
  postcode: {
    flex: 1,
  },
});

export default AddressEditScreen;