import os



pairs = [
    # Greetings
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey! How's it going?", "Hi! How can I help you today?"]),
    (r"howdy", ["Howdy! What's up?", "Howdy! How's everything going?"]),
    (r"good morning", ["Good morning! How are you today?", "Good morning! Ready for a great day?"]),
    (r"good afternoon", ["Good afternoon! How's your day going?", "Good afternoon! What's new?"]),
    (r"good evening", ["Good evening! How was your day?", "Good evening! How can I assist you today?"]),

    # Responses to how the user is feeling
(r"thank you|thanks|thanks a lot|thanks so much|many thanks|thanks a million|i appreciate it|thanks for your help|much obliged|thanks heaps",
     ["You're welcome!", "No problem.", "Anytime!", "Happy to help.", "My pleasure.", "Don't mention it.", "Glad I could help.", "That's what I'm here for.", "Of course.", "No worries."]),



    (r"how are you\??",
     ["I'm doing great, thank you!", "I'm good, thanks for asking!", "I'm doing well! How about you?"]),
    (r"i'm fine", ["Glad to hear it!", "That's great!", "Awesome, glad you're doing well!"]),
    (r"i'm not feeling great",
     ["Oh no, I'm sorry to hear that. What's wrong?", "I'm here for you if you need to talk!"]),
    (r"i'm tired", ["I understand. It’s been a long day. Want to talk about it?", "Rest is important. Take it easy!"]),
    (r"i'm feeling happy", ["That's wonderful! What’s making you so happy today?",
                            "I'm so glad to hear that! Keep that positive energy going!"]),

    # Asking about the bot
    (r"what is your name\??", ["I am an AI .I don’t have a personal name, but you can call me ZackBird."]),
    (r"tell me your name\??", ["I am an AI .I don’t have a personal name, but you can call me ZackBird."]),
    (
    r"who are you\??", ["I’m an AI chatbot designed to help with various questions.", "I am your friendly assistant!"]),
    (r"what can you do\??", ["I can help answer questions, give suggestions, or just chat with you!",
                             "I can assist you with information, play games, or have a conversation!,plot graphs, solve linear, quadratic equations etc."]),

    # Asking the bot about its capabilities
    (r"are you smart\??",
     ["I try my best! I have a lot of information to share.", "I like to think I'm pretty smart! How about you?"]),
    (r"can you think\??", ["I don't think like humans, but I can process a lot of information!",
                           "I can understand and respond to questions, but I don't have emotions or consciousness."]),
# Study-related Responses

# Basic Arithmetic Operations
(r"(what are|what do you mean by) basic arithmetic operations\??", [
    "The basic arithmetic operations are addition, subtraction, multiplication, and division. These are the foundation of all other mathematical operations.",
    "In simple terms, addition combines numbers, subtraction finds the difference, multiplication scales numbers, and division splits numbers into equal parts."
]),

(r"(how|in what way) do you perform basic arithmetic operations\??", [
    "Basic arithmetic operations involve adding, subtracting, multiplying, and dividing numbers. For example, 5 + 3 = 8 or 10 ÷ 2 = 5.",
    "You can use calculators or do these operations manually. For more complex problems, you may use order of operations (PEMDAS)."
]),

# Solving Quadratic Equations
(r"how do I (solve|find the solutions of) a quadratic equation\??", [
    "To solve a quadratic equation, use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a, where 'a', 'b', and 'c' are the coefficients from the equation ax² + bx + c = 0.",
    "You can also factor the quadratic equation, complete the square, or use the quadratic formula, depending on the form of the equation. You can write your equation exactly as you would say it, or use the quadratic formula to solve it.Eg. 2x^2+6x-6"
]),

(r"what is the method for solving a quadratic equation\??", [
    "The most common method is using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. Alternatively, you could try factoring or completing the square if the equation is factorable.",
    "Would you like to see an example? I can help you solve a quadratic equation step by step."
]),

# Pythagorean Theorem
(r"(what|could you explain) is the Pythagorean theorem\??", [
    "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. Formula: a² + b² = c².",
    "This theorem is essential for solving problems involving right-angled triangles and is used in geometry and trigonometry."
]),

(r"how do I use the Pythagorean theorem\??", [
    "To use the Pythagorean theorem, you need the lengths of two sides of a right triangle. Use the formula a² + b² = c² to find the third side.",
    "If you know the lengths of the legs (a and b), you can solve for the hypotenuse (c), or vice versa."
]),

# Area of a Triangle
(r"(how do|what's the method for) calculating the area of a triangle\??", [
    "The area of a triangle is calculated using the formula: Area = ½ * base * height. The base is any side of the triangle, and the height is the perpendicular distance from that base to the opposite vertex.",
    "For special triangles like equilateral triangles, you can use a different formula, such as Area = (√3 / 4) * side², where 'side' is the length of a side."
]),

(r"how can I (find|calculate) the area of a triangle\??", [
    "To find the area, multiply the base of the triangle by its height and divide by 2. This gives you the amount of space inside the triangle.",
    "If the triangle is equilateral, the formula becomes Area = (√3 / 4) * side²."
]),


# Prime Numbers
(r"(what|could you tell me) what a prime number is\??", [
    "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples include 2, 3, 5, 7, and 11.",
    "Prime numbers are fundamental in number theory and are used in cryptography, among other applications."
]),

(r"how do I (identify|find) a prime number\??", [
    "To identify a prime number, check if it has only two factors: 1 and itself. If it can be divided evenly by any other number, it is not prime.",
    "Examples of prime numbers are 2, 3, 5, 7, 11, and 13."
]),
(r"how can I (find|calculate) the area of a rectangle\??", [
    "To find the area of a rectangle, multiply the width of the rectangle by its height a. This gives you the amount of space inside the rectangle.",

]),
(r"(how do|what's the method for) calculating the area of a triangle\??", [
    "The area of a rectangle is calculated using the formula: Area = width * height. The base is any side of the rectangle,"
]),


# Permutations and Combinations
(r"what's the difference between (permutations|combinations)\??", [
    "Permutations refer to the arrangement of objects in a specific order, while combinations refer to the selection of objects without regard to the order.",
    "In permutations, the order matters; in combinations, it does not."
]),

(r"can you explain (permutations|combinations)\??", [
    "Permutations and combinations are methods of counting. Permutations deal with ordered arrangements, while combinations deal with selections where order does not matter.",
    "The formula for permutations is P(n, r) = n! / (n - r)!, and for combinations, it's C(n, r) = n! / [r!(n - r)!]."
]),

# Derivatives and Integration
(r"what is the (derivative|rate of change) of a function\??", [
    "The derivative of a function represents its rate of change or the slope of the tangent line at any given point.",
    "To differentiate a function, use rules like the power rule, product rule, quotient rule, or chain rule, depending on the form of the function."
]),

(r"how do I (integrate|find the integral of) a function\??", [
    "To integrate a function, you use integration rules or formulas such as the power rule, substitution rule, or integration by parts.",
    "The basic formula for integration is ∫f(x) dx = F(x) + C, where F(x) is the antiderivative of f(x), and C is the constant of integration."
]),

# Vectors and Matrices
(r"what are vectors used for in mathematics\??", [
    "A vector is a quantity that has both magnitude and direction. Vectors are used in physics, engineering, and mathematics to represent quantities such as velocity and force.",
    "In mathematics, vectors can be added, subtracted, and multiplied by scalars."
]),

(r"how do I (perform operations|do math with) matrices\??", [
    "You can perform operations on matrices like addition, subtraction, multiplication, and finding determinants or inverses.",
    "Matrix multiplication involves multiplying the rows of the first matrix by the columns of the second matrix."
]),

# Probability
(r"how do I calculate probability\??", [
    "Probability is calculated as the ratio of the number of favorable outcomes to the total number of possible outcomes in a sample space.",
    "It ranges from 0 (impossible) to 1 (certain). For example, the probability of rolling a 3 on a fair six-sided die is 1/6."
]),

(r"what is the (probability|likelihood) of an event\??", [
    "The probability of an event is the chance that it will occur. It's calculated by dividing the number of successful outcomes by the total number of outcomes.",
    "For example, flipping a coin has a 50% probability of landing heads."
]),

# Graphing Linear Equations
(r"(how do|what's the method for) graphing a linear equation\??", [
    "To graph a linear equation, first rewrite it in slope-intercept form (y = mx + b), where 'm' is the slope and 'b' is the y-intercept.",
    "Plot the y-intercept (0, b) on the graph and use the slope to determine the rise over run. Draw a line through these points to represent the equation. You can write the linear equation, and I will graph it for you."
]),

(r"(can|how can) I graph a line from a linear equation\??", [
    "You can graph a linear equation by finding two points. Start with the y-intercept, and then use the slope to find another point. Connect the points with a straight line.",
    "If you provide the equation, I can help graph it for you."
]),

(r"how can I improve my study habits\??", [
    "To improve your study habits, try setting a consistent schedule, breaking down tasks into smaller parts, and staying organized with notes and study materials.",
    "Good study habits include staying focused, eliminating distractions, taking regular breaks, and reviewing material frequently."
]),

# Business-related Responses
(r"what is business\??", [
    "Business is the activity of making one's living or making money by producing or buying and selling products (such as goods and services).",
    "Business refers to an organization or enterprising entity engaged in commercial, industrial, or professional activities."
]),

(r"how to start a business\??", [
    "Starting a business involves researching your market, creating a business plan, and securing financing. Would you like to know more about any step?",
    "To start a business, you’ll need a solid idea, a business plan, funding, and a strategy to make your business successful."
]),

(r"what are the key elements of a business\??", [
    "The key elements include a strong business idea, a target market, a business model, financial resources, and a good marketing strategy.",
    "A business needs a clear value proposition, a customer base, revenue streams, operations, and a marketing strategy."
]),

(r"how do i grow my business\??", [
    "To grow a business, focus on improving customer satisfaction, scaling operations, enhancing marketing efforts, and diversifying your product offerings.",
    "Growing a business requires strategic planning, effective marketing, expanding your customer base, and continuous improvement."
]),

(r"what is entrepreneurship\??", [
    "Entrepreneurship is the process of starting and running a new business in order to make a profit, often involving innovation and risk-taking.",
    "Entrepreneurship involves identifying opportunities, taking risks, and using resources to build a business that creates value."
]),

(r"how to manage a business\??", [
    "Managing a business involves planning, organizing resources, leading employees, and controlling financial aspects. It’s essential to keep track of performance and adapt strategies.",
    "Good management includes setting goals, monitoring progress, motivating your team, and adapting to changing market conditions."
]),

(r"what is a business plan\??", [
    "A business plan is a detailed document that outlines the goals, strategy, and financial projections of your business.",
    "A business plan serves as a roadmap for your business, helping you define your objectives and the steps to reach them."
]),

(r"what is a marketing strategy\??", [
    "A marketing strategy is a plan for promoting and selling your products or services, focusing on target customers and the methods to reach them.",
    "A good marketing strategy includes understanding your target audience, choosing the right channels, and setting clear objectives."
]),

(r"how to attract customers\??", [
    "To attract customers, focus on offering value, marketing effectively, building a strong brand, and engaging with potential clients through various channels.",
    "Attracting customers requires understanding their needs, delivering quality products or services, and using the right marketing techniques."
]),

(r"what is customer service\??", [
    "Customer service is the support you offer to your customers before, during, and after a purchase, ensuring their satisfaction.",
    "Good customer service helps build relationships, increase customer loyalty, and encourage repeat business."
]),

(r"how do i finance my business\??", [
    "You can finance your business through personal savings, loans, investors, crowdfunding, or venture capital, depending on your needs and business stage.",
    "Financing options for your business include applying for a business loan, seeking angel investors, or using crowdfunding platforms."
]),

(r"what is business growth\??", [
    "Business growth refers to the process of expanding a company's market reach, increasing revenue, or scaling operations.",
    "Growth in business is about increasing profits, improving efficiency, expanding the customer base, or entering new markets."
]),

(r"what is a startup\??", [
    "A startup is a newly established business that is typically focused on developing a unique product or service and is usually in the early stages of growth.",
    "Startups are often characterized by innovation and the need for funding to grow quickly and scale their operations."
]),

(r"what is e-commerce\??", [
    "E-commerce is the buying and selling of goods or services using the internet, along with the transfer of money and data to execute these transactions.",
    "E-commerce allows businesses to reach a global customer base, making transactions more convenient and accessible."
]),

(r"how do i increase sales\??", [
    "To increase sales, improve your marketing, enhance customer experience, offer promotions, and create strong relationships with your customers.",
    "Focus on increasing customer retention, offering valuable products, and finding new markets to increase sales."
]),

(r"what is a target market\??", [
    "A target market is a specific group of customers that a business aims to serve with its products or services, typically based on demographics, behavior, or needs.",
    "Identifying your target market helps you tailor your marketing efforts and product offerings to meet their specific needs."
]),

(r"how do i manage business risks\??", [
    "Managing business risks involves identifying potential risks, assessing their impact, and implementing strategies to mitigate them, such as diversification, insurance, and contingency planning.",
    "Effective risk management requires proactive planning, monitoring, and adjusting your business strategies to minimize potential losses."
]),

(r"what is business ethics\??", [
    "Business ethics refers to the moral principles and standards that guide behavior in the business world, ensuring fairness, transparency, and responsibility.",
    "Ethical businesses prioritize honesty, accountability, and respect for customers, employees, and the environment."
]),

(r"what is leadership in business\??", [
    "Leadership in business is about guiding and motivating a team to achieve company goals, inspiring others, and making strategic decisions.",
    "Good leadership involves setting a clear vision, making informed decisions, and fostering a positive and productive work environment."
]),

(r"how do i manage employees\??", [
    "Managing employees effectively requires clear communication, providing feedback, setting expectations, and offering opportunities for growth and development.",
    "Good employee management involves understanding their needs, providing support, and recognizing their achievements."
]),


(r"how to stay focused while studying\??", [
    "Staying focused can be achieved by setting specific goals, eliminating distractions (like your phone), and taking breaks when needed to refresh your mind.",
    "Try studying in a quiet environment, using the Pomodoro technique, and taking short breaks to maintain focus."
]),

(r"how do I manage my time better for studying\??", [
    "Time management can be improved by creating a study schedule, prioritizing tasks, and breaking large tasks into smaller, manageable chunks.",
    "Use time management techniques like the Pomodoro method or time blocking to structure your study sessions effectively."
]),

(r"what are the best study techniques\??", [
    "Some of the best study techniques include active recall, spaced repetition, summarizing material in your own words, and teaching the material to someone else.",
    "Techniques like mind mapping, note-taking, and solving practice problems help reinforce learning and make studying more effective."
]),

(r"how do I stay motivated to study\??", [
    "Set small, achievable goals and reward yourself when you accomplish them. Staying motivated is easier when you can see progress.",
    "Keep reminding yourself of your long-term goals and the benefits of your studies. Break your tasks into smaller parts to avoid feeling overwhelmed."
]),

(r"how to study for exams\??", [
    "For exams, focus on understanding key concepts, practice with past papers, and revise consistently rather than cramming at the last minute.",
    "Make a study plan, use active recall, and go over practice questions to test your knowledge before the exam."
]),

(r"how do I deal with exam stress\??", [
    "Take deep breaths, stay organized, and avoid last-minute cramming. Regular exercise and good sleep can also help reduce stress.",
    "It's normal to feel stressed before an exam. Try to stay calm, prepare in advance, and take breaks to relieve pressure."
]),

(r"how can I improve my memory for studying\??", [
    "Improving memory can be done by using techniques like spaced repetition, mnemonics, and visualizing information.",
    "Try associating new information with something you already know, and use active recall to help reinforce what you've learned."
]),

(r"how do I balance studies and social life\??", [
    "Balance comes with good time management. Plan your study sessions and make sure you allocate time for relaxation and socializing.",
    "It's important to have a balanced routine. Allocate time for both studying and social activities, ensuring neither takes over your schedule."
]),

(r"how do I stay organized with my studies\??", [
    "Staying organized involves keeping track of assignments, making to-do lists, using a planner, and regularly reviewing your study materials.",
    "Use digital tools like Google Calendar or a notebook to organize your study schedule and deadlines."
]),

(r"how do I avoid procrastination\??", [
    "To avoid procrastination, try breaking tasks into smaller steps, setting specific deadlines, and using techniques like the Pomodoro method to stay on track.",
    "Start with simple tasks to build momentum. Eliminate distractions, set clear goals, and hold yourself accountable."
]),

(r"how do I take better notes\??", [
    "Take notes in your own words, organize them with headings and bullet points, and use diagrams or mind maps to visualize concepts.",
    "Try different note-taking techniques like the Cornell Method or outlining, and always review your notes regularly."
]),

(r"how to improve my writing skills for essays\??", [
    "To improve your writing skills, practice regularly, read more essays, and focus on clarity, structure, and argumentation.",
    "Start by organizing your thoughts before writing, use strong thesis statements, and support your arguments with evidence and examples."
]),

(r"how can I stay disciplined in my studies\??", [
    "Discipline comes from creating a routine, setting clear goals, and sticking to your schedule even when it gets tough.",
    "Stay disciplined by creating a study plan, minimizing distractions, and holding yourself accountable for your academic goals."
]),

(r"how to study effectively\??", [
    "Study effectively by actively engaging with the material, summarizing key points, and practicing what you've learned.",
    "Use active learning techniques, such as self-quizzing, explaining concepts to others, and applying knowledge through real-life examples."
]),

(r"what should I do if I don’t understand something\??", [
    "If you're stuck, try breaking the problem into smaller pieces, look for additional resources, or ask for help from a teacher or fellow student.",
    "Don’t hesitate to seek help. Discuss the problem with peers, watch tutorial videos, or go over the material again to gain a better understanding."
]),

(r"how can I improve my reading speed\??", [
    "To improve reading speed, practice reading regularly, eliminate subvocalization, and try techniques like skimming and scanning for key points.",
    "Try reading in chunks and focus on improving your comprehension while reading faster."
]),

(r"how can I focus better in class\??", [
    "Stay engaged by taking notes, asking questions, and participating in discussions. Minimize distractions by putting your phone away.",
    "Focus by setting goals for each class, listening attentively, and taking notes to stay engaged with the material."
]),

(r"how do I find the best study environment\??", [
    "Find a quiet space free from distractions, with good lighting and a comfortable seating arrangement. Some people prefer libraries, while others need a bit of background noise.",
    "Experiment with different study locations to see what works best for you. Some prefer silence, while others benefit from background music or a coffee shop environment."
]),

(r"how do I deal with burnout from studying\??", [
    "Take breaks and make time for hobbies or social activities. Burnout happens when you push yourself too hard, so it’s important to rest and recharge.",
    "If you feel burned out, take time off from studying, practice relaxation techniques, and consider adjusting your workload to avoid over-exertion."
]),


    # Asking about the weather
    (r"what's the weather like\??", ["I'm not sure about the weather right now, but I can help you find it online!",
                                     "I can't check the weather, but you can easily find it on a weather app!"]),
    (r"is it going to rain\??", ["I can't check the weather for you, but you can check a weather website or app!",
                                 "I recommend checking your local weather app for rain updates!"]),

# Mental Health & Psychology Responses
(r"how do I manage stress\??", [
    "Managing stress involves recognizing the signs, practicing deep breathing, exercising regularly, and taking time for relaxation and hobbies.",
    "Try breaking tasks into smaller steps, taking breaks, and engaging in stress-reducing activities like meditation or yoga."
]),

(r"what are the signs of anxiety\??", [
    "Common signs of anxiety include excessive worry, restlessness, difficulty concentrating, physical symptoms like a racing heart, and avoiding situations that cause fear.",
    "Anxiety can manifest as feeling on edge, having trouble sleeping, or experiencing sudden bursts of nervousness or tension."
]),

(r"how do I deal with anxiety\??", [
    "Dealing with anxiety can include practicing relaxation techniques like deep breathing, mindfulness meditation, or exercising to reduce tension.",
    "It’s important to identify the sources of your anxiety, try cognitive-behavioral techniques, and seek professional help if needed."
]),

(r"what are the symptoms of depression\??", [
    "Symptoms of depression include persistent sadness, loss of interest in activities, fatigue, trouble sleeping, changes in appetite, and feelings of hopelessness.",
    "Depression can affect your energy levels, cause difficulty concentrating, and may lead to withdrawal from social activities."
]),

(r"how can I cope with depression\??", [
    "Coping with depression involves seeking support from loved ones, practicing self-care, setting small achievable goals, and seeking professional help when necessary.",
    "You might consider engaging in activities you enjoy, maintaining a routine, and talking to someone you trust to help manage depressive feelings."
]),

(r"what is a panic attack\??", [
    "A panic attack is a sudden episode of intense fear or discomfort, often accompanied by physical symptoms like a racing heart, difficulty breathing, and dizziness.",
    "Panic attacks are overwhelming and can feel like you're losing control, but they are temporary and can be managed with coping strategies."
]),

(r"how can I overcome a panic attack\??", [
    "During a panic attack, try to focus on your breathing by taking slow, deep breaths. Ground yourself by focusing on your surroundings and reminding yourself that the attack is temporary.",
    "A helpful technique is the 5-4-3-2-1 grounding exercise: identify 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
]),

(r"what is self-care\??", [
    "Self-care is the practice of taking time to care for your mental, emotional, and physical well-being. It can involve activities like meditation, exercising, reading, or simply resting.",
    "Self-care means making time for yourself, doing activities that reduce stress, and prioritizing your mental health."
]),

(r"how do I build self-esteem\??", [
    "Building self-esteem starts with positive self-talk, setting achievable goals, celebrating your accomplishments, and surrounding yourself with supportive people.",
    "Try focusing on your strengths, acknowledging your achievements, and learning to be kind to yourself, even when things don't go as planned."
]),

(r"how can I help someone with depression\??", [
    "If someone is struggling with depression, listen to them without judgment, encourage them to seek professional help, and offer to support them in any way you can.",
    "You can help by being a supportive listener, suggesting activities they enjoy, and reminding them that it's okay to seek help from a therapist or counselor."
]),

(r"how can I reduce negative thoughts\??", [
    "To reduce negative thoughts, practice challenging those thoughts by replacing them with positive or balanced alternatives. Cognitive-behavioral techniques can be helpful.",
    "Mindfulness and meditation can help you become more aware of negative thoughts and create space to choose more positive or realistic thinking."
]),

(r"what is mindfulness\??", [
    "Mindfulness is the practice of focusing on the present moment, accepting it without judgment, and being aware of your thoughts, feelings, and surroundings.",
    "It involves paying attention to the current moment, practicing self-awareness, and being non-judgmental toward your emotions and experiences."
]),

(r"how do I practice mindfulness\??", [
    "To practice mindfulness, try to focus on your breath, observe your surroundings, and allow yourself to be present in the moment without distractions.",
    "You can practice mindfulness by doing a body scan, engaging in mindful breathing, or paying attention to your senses while walking or eating."
]),

(r"how do I deal with social anxiety\??", [
    "Dealing with social anxiety involves gradual exposure to social situations, learning relaxation techniques, and challenging negative beliefs about social interactions.",
    "It can help to focus on others rather than yourself during social events and practice self-compassion instead of criticizing yourself."
]),

(r"what is emotional intelligence\??", [
    "Emotional intelligence (EI) is the ability to recognize, understand, and manage your own emotions, as well as the ability to recognize, understand, and influence the emotions of others.",
    "Developing EI involves practicing empathy, improving emotional awareness, and learning how to regulate your emotional responses in different situations."
]),

(r"how do I improve emotional intelligence\??", [
    "You can improve emotional intelligence by practicing self-awareness, developing empathy, learning to manage your emotions, and improving your communication skills.",
    "Start by reflecting on your own emotions, listening actively to others, and learning how to respond thoughtfully to emotional situations."
]),

(r"how do I deal with feelings of loneliness\??", [
    "To deal with loneliness, try reaching out to friends or family, participating in group activities, or finding new hobbies that connect you with others.",
    "Consider volunteering or joining communities with shared interests. It’s important to remind yourself that loneliness is a common experience, and it’s okay to seek support."
]),

(r"how do I cope with grief\??", [
    "Coping with grief involves allowing yourself to feel and process your emotions, talking to supportive people, and finding healthy outlets for your sadness.",
    "Grief takes time. Seek support from loved ones or consider talking to a counselor to help you navigate through the grieving process."
]),

(r"what is therapy\??", [
    "Therapy is a treatment that involves talking to a trained mental health professional to address emotional and psychological challenges, such as anxiety, depression, or relationship issues.",
    "Therapy provides a safe space to explore your thoughts, feelings, and behaviors, helping you develop coping strategies and understand yourself better."
]),

(r"how can therapy help me\??", [
    "Therapy can help you by offering tools to manage stress, anxiety, depression, and other mental health challenges, as well as providing a non-judgmental space to express yourself.",
    "Through therapy, you can gain insights into your emotions, learn healthier coping strategies, and work toward personal growth and healing."
]),



    # Asking about hobbies and interests
    (r"what do you like to do\??",
     ["I enjoy chatting with people like you! I don’t have hobbies like humans do, but I love learning new things.",
      "I like answering questions and helping out! What about you?"]),
    (r"what are your hobbies\??", ["I don't have hobbies like humans, but I enjoy processing information and chatting!",
                                   "I don't have personal hobbies, but I can help you find yours!"]),
    (r"do you have any interests\??", ["My primary interest is assisting you! What are your interests?",
                                       "I'm really interested in helping and learning! How about you?"]),

    # Favorite things
    (r"what's your favorite color\??", ["I don't have a favorite color, but I think all colors are beautiful!",
                                        "I don’t have preferences, but I think blue is a lovely color!"]),
    (r"what's your favorite food\??",
     ["I don’t eat, but I think pizza is popular!", "I don’t have a favorite food, but burgers are pretty tasty!"]),
    (r"what's your favorite movie\??", ["I don’t watch movies, but I’ve heard 'The Matrix' is a cool one!",
                                        "I don’t have a favorite, but I think sci-fi movies are exciting!"]),

    # More emotional responses
    (r"i'm sad",
     ["I’m really sorry to hear that. Want to talk about it?", "It's okay to feel sad sometimes. I’m here for you."]),
    (r"i'm angry", ["I'm sorry you're feeling that way. Want to talk about what's bothering you?",
                    "It can be tough to be angry. Maybe taking a few deep breaths could help."]),
(r"i am sad",
     ["I’m really sorry to hear that. Want to talk about it?", "It's okay to feel sad sometimes. I’m here for you."]),


    # Fun questions
    (r"tell me a joke", ["Why don’t skeletons fight each other? They don’t have the guts!",
                         "Why did the computer break up with the internet? There was no connection!"]),
    (r"tell me something interesting", [
        "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient tombs that are over 3000 years old!",
        "Here’s something cool: Bananas are berries, but strawberries aren’t!"]),

    # Asking about the future
    (r"what will happen tomorrow\??", ["I can't predict the future, but I hope tomorrow is a great day for you!",
                                       "I can’t predict the future, but I believe tomorrow has potential!"]),
    (r"what will the weather be like tomorrow\??", ["I recommend checking your local weather app to find out!",
                                                    "I can’t check the weather, but you can easily get the forecast on your phone!"]),
    ("i am sickness", [
        "It seems you're not feeling well. Could you describe your symptoms? For example: 'I have a headache and fever.'"
    ]),
    ("i am sick", [
        "I'm sorry to hear that. Could you let me know your symptoms so I can help you better?"
    ]),
    ("i feel sick", [
        "Thanks for sharing. Could you please describe what you're experiencing, like cough, fever, or fatigue?"
    ]),
    ("i have sickness", [
        "Got it. Can you tell me more? For example: 'I have nausea and body aches.'"
    ]),
    ("feeling unwell", [
        "I'm here to help. What symptoms are you experiencing?"
    ]),
    ("not feeling good", [
        "Sorry you're not feeling well. Could you list your symptoms like 'cold, cough, or sore throat'?"
    ]),
    ("i don't feel good", [
        "I understand. Could you share your symptoms so I can assist properly?"
    ]),
    (r"what can you do\??", [
        "I can help answer questions, give suggestions, assist with health-related queries, and even generate charts and equations!",
        "I can provide useful information, assist with math problems, and help you with health-related questions, just to name a few!"
    ]),

    ("i'm frustrated", [
        "I'm really sorry you're feeling frustrated. Want to talk about it?",
        "It’s okay to feel frustrated sometimes. Let me know if there’s something I can do to help."
    ]),
    ("i'm confused", [
        "I'm here to help! What’s got you feeling confused?",
        "It’s okay to feel confused. Let’s break it down together!"
    ]),
    ("i feel anxious", [
        "I'm sorry to hear that. Are you feeling anxious about something specific?",
        "Anxiety can be tough. Take deep breaths and let me know how I can assist you."
    ]),
# Business-related Responses
(r"what is business\??", [
    "Business is the activity of making one's living or making money by producing or buying and selling products (such as goods and services).",
    "Business refers to an organization or enterprising entity engaged in commercial, industrial, or professional activities."
]),

(r"how to start a business\??", [
    "Starting a business involves researching your market, creating a business plan, and securing financing. Would you like to know more about any step?",
    "To start a business, you’ll need a solid idea, a business plan, funding, and a strategy to make your business successful."
]),

(r"what are the key elements of a business\??", [
    "The key elements include a strong business idea, a target market, a business model, financial resources, and a good marketing strategy.",
    "A business needs a clear value proposition, a customer base, revenue streams, operations, and a marketing strategy."
]),

(r"how do i grow my business\??", [
    "To grow a business, focus on improving customer satisfaction, scaling operations, enhancing marketing efforts, and diversifying your product offerings.",
    "Growing a business requires strategic planning, effective marketing, expanding your customer base, and continuous improvement."
]),

(r"what is entrepreneurship\??", [
    "Entrepreneurship is the process of starting and running a new business in order to make a profit, often involving innovation and risk-taking.",
    "Entrepreneurship involves identifying opportunities, taking risks, and using resources to build a business that creates value."
]),

(r"how to manage a business\??", [
    "Managing a business involves planning, organizing resources, leading employees, and controlling financial aspects. It’s essential to keep track of performance and adapt strategies.",
    "Good management includes setting goals, monitoring progress, motivating your team, and adapting to changing market conditions."
]),

(r"what is a business plan\??", [
    "A business plan is a detailed document that outlines the goals, strategy, and financial projections of your business.",
    "A business plan serves as a roadmap for your business, helping you define your objectives and the steps to reach them."
]),

(r"what is a marketing strategy\??", [
    "A marketing strategy is a plan for promoting and selling your products or services, focusing on target customers and the methods to reach them.",
    "A good marketing strategy includes understanding your target audience, choosing the right channels, and setting clear objectives."
]),

(r"how to attract customers\??", [
    "To attract customers, focus on offering value, marketing effectively, building a strong brand, and engaging with potential clients through various channels.",
    "Attracting customers requires understanding their needs, delivering quality products or services, and using the right marketing techniques."
]),

(r"what is customer service\??", [
    "Customer service is the support you offer to your customers before, during, and after a purchase, ensuring their satisfaction.",
    "Good customer service helps build relationships, increase customer loyalty, and encourage repeat business."
]),

(r"how do i finance my business\??", [
    "You can finance your business through personal savings, loans, investors, crowdfunding, or venture capital, depending on your needs and business stage.",
    "Financing options for your business include applying for a business loan, seeking angel investors, or using crowdfunding platforms."
]),

(r"what is business growth\??", [
    "Business growth refers to the process of expanding a company's market reach, increasing revenue, or scaling operations.",
    "Growth in business is about increasing profits, improving efficiency, expanding the customer base, or entering new markets."
]),

(r"what is a startup\??", [
    "A startup is a newly established business that is typically focused on developing a unique product or service and is usually in the early stages of growth.",
    "Startups are often characterized by innovation and the need for funding to grow quickly and scale their operations."
]),

(r"what is e-commerce\??", [
    "E-commerce is the buying and selling of goods or services using the internet, along with the transfer of money and data to execute these transactions.",
    "E-commerce allows businesses to reach a global customer base, making transactions more convenient and accessible."
]),

(r"how do i increase sales\??", [
    "To increase sales, improve your marketing, enhance customer experience, offer promotions, and create strong relationships with your customers.",
    "Focus on increasing customer retention, offering valuable products, and finding new markets to increase sales."
]),

(r"what is a target market\??", [
    "A target market is a specific group of customers that a business aims to serve with its products or services, typically based on demographics, behavior, or needs.",
    "Identifying your target market helps you tailor your marketing efforts and product offerings to meet their specific needs."
]),

(r"how do i manage business risks\??", [
    "Managing business risks involves identifying potential risks, assessing their impact, and implementing strategies to mitigate them, such as diversification, insurance, and contingency planning.",
    "Effective risk management requires proactive planning, monitoring, and adjusting your business strategies to minimize potential losses."
]),

(r"what is business ethics\??", [
    "Business ethics refers to the moral principles and standards that guide behavior in the business world, ensuring fairness, transparency, and responsibility.",
    "Ethical businesses prioritize honesty, accountability, and respect for customers, employees, and the environment."
]),

(r"what is leadership in business\??", [
    "Leadership in business is about guiding and motivating a team to achieve company goals, inspiring others, and making strategic decisions.",
    "Good leadership involves setting a clear vision, making informed decisions, and fostering a positive and productive work environment."
]),

(r"how do i manage employees\??", [
    "Managing employees effectively requires clear communication, providing feedback, setting expectations, and offering opportunities for growth and development.",
    "Good employee management involves understanding their needs, providing support, and recognizing their achievements."
]),





# Laughter Responses
(r"haha|lol|lmao|rofl", [
    "You're making me laugh too! 😂",
    "Haha, you're funny! 😄",
    "LOL! That cracked me up!",
    "You’ve got a great sense of humor! 😂",
    "Haha, good one! 😆"
]),

(r"i'm laughing", [
    "Haha, that's awesome! Keep it up!",
    "Glad you're having fun! 😄",
    "Laughter is the best medicine! 😆",
    "Your laugh is contagious! 😂"
]),

(r"that's funny", [
    "I'm glad you think so! 😄",
    "I knew you'd like that! 😂",
    "It’s always great to share a laugh! 😆",
    "Humor makes everything better, right? 😂"
]),

(r"you made me laugh", [
    "Mission accomplished! 😆",
    "Happy to make you smile! 😂",
    "Laughter is the best! 😄",
    "I love making people laugh! 😄"
]),

(r"you're funny", [
    "Haha, thanks! I try my best! 😄",
    "Glad you think so! 😆",
    "You’ve got some good taste in humor! 😆",
    "I appreciate that! Let’s keep the laughs going! 😂"
]),

    # Asking for help or advice
    (
        r"can you help me\??",
        ["Of course! What do you need help with?", "I'm here to help! What can I assist you with?"]),
    (r"i need advice", ["I’d be happy to help! What do you need advice on?", "I can try to help! What’s going on?"]),

    # Saying goodbye
    (r"bye|goodbye", ["Bye! Have a wonderful day!", "Goodbye! Take care and talk to you soon!"]),
    (r"see you later", ["See you later! Take care!", "Catch you later! Have a good one!"]),
    (r"talk to you soon", ["Talk to you soon! I’ll be here if you need me!", "Looking forward to our next chat!"]),
]

def chat_data():
    return pairs




