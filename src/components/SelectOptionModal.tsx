import React from 'react';
import { Modal, View, Pressable, StyleSheet, FlatList } from 'react-native';
import CustomText from './CustomText.tsx';

interface SelectOptionModalProps {
  visible: boolean;
  options: string[];
  onSelect: (option: string) => void;
  onClose: () => void;
}

export default function SelectOptionModal({ visible, options, onSelect, onClose }: SelectOptionModalProps) {
  return (
    <Modal visible={visible} transparent animationType="slide">
      <Pressable style={styles.overlay} onPress={onClose} />
      <View style={styles.modalContainer}>
        <FlatList
          data={options}
          keyExtractor={(item) => item}
          renderItem={({ item }) => (
            <Pressable style={styles.option} onPress={() => onSelect(item)}>
              <CustomText style={styles.optionText}>{item}</CustomText>
            </Pressable>
          )}
        />
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: '#00000088',
  },
  modalContainer: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    backgroundColor: 'white',
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    padding: 16,
    maxHeight: '50%',
  },
  option: {
    padding: 12,
    borderBottomColor: '#ddd',
    borderBottomWidth: 1,
  },
  optionText: {
    fontSize: 18,
  },
});
