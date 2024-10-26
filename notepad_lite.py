import streamlit as st
import base64

def notepad():
    st.markdown('<div class="section-title">📝 Notepad</div>', unsafe_allow_html=True)

    # Toggle visibility of the notepad
    if 'notepad_open' not in st.session_state:
        st.session_state.notepad_open = True

    if st.button("🗙 Close Notepad"):
        st.session_state.notepad_open = False

    if st.session_state.notepad_open:
        # Create a text area for the notepad
        text = st.text_area("Write your notes here:", height=300)

        # Button to save notes to a file
        if st.button("Save Notes to PC"):
            if text:
                save_file(text)
            else:
                st.warning("Please write something to save.")

        # Button to clear the text area
        if st.button("Clear Notes"):
            st.text_area("Write your notes here:", height=300, value="")  # Clear the text area

def save_file(text):
    # Encode the text to a downloadable format
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="notes.txt">Click here to download your notes</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("Your notes are ready for download!")

# The notepad function can be called from your main app (home.py).
if __name__ == "__main__":
    notepad()
