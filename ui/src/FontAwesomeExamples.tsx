import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';

// Import solid icons
import { 
  faHome, 
  faUser, 
  faCog, 
  faHeart, 
  faStar,
  faSearch,
  faBell,
  faEnvelope,
  faPhone,
  faCamera,
  faVideo,
  faDownload,
  faUpload,
  faEdit,
  faTrash,
  faSave,
  faPlus,
  faMinus,
  faCheck,
  faTimes,
  faArrowLeft,
  faArrowRight,
  faArrowUp,
  faArrowDown
} from '@fortawesome/free-solid-svg-icons';

// Import regular icons
import { 
  faHeart as faHeartRegular,
  faStar as faStarRegular,
  faUser as faUserRegular,
  faEnvelope as faEnvelopeRegular,
  faBell as faBellRegular
} from '@fortawesome/free-regular-svg-icons';

// Import brand icons
import { 
  faGithub, 
  faTwitter, 
  faFacebook, 
  faInstagram, 
  faLinkedin,
  faYoutube,
  faGoogle,
  faApple,
  faMicrosoft,
  faAmazon
} from '@fortawesome/free-brands-svg-icons';

const FontAwesomeExamples: React.FC = () => {
  return (
    <div className="p-8 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-8 text-center">FontAwesome Icons Examples</h1>
      
      {/* Basic Usage */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Basic Usage</h2>
        <div className="grid grid-cols-4 gap-4">
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faHome} />
            <span>Home</span>
          </div>
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faUser} />
            <span>User</span>
          </div>
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faCog} />
            <span>Settings</span>
          </div>
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faSearch} />
            <span>Search</span>
          </div>
        </div>
      </section>

      {/* Different Sizes */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Different Sizes</h2>
        <div className="flex items-center gap-4">
          <FontAwesomeIcon icon={faStar} size="xs" />
          <FontAwesomeIcon icon={faStar} size="sm" />
          <FontAwesomeIcon icon={faStar} size="lg" />
          <FontAwesomeIcon icon={faStar} size="xl" />
          <FontAwesomeIcon icon={faStar} size="2xl" />
          <FontAwesomeIcon icon={faStar} size="3xl" />
        </div>
      </section>

      {/* Different Colors */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Different Colors</h2>
        <div className="flex items-center gap-4">
          <FontAwesomeIcon icon={faHeart} style={{ color: 'red' }} />
          <FontAwesomeIcon icon={faHeart} style={{ color: 'blue' }} />
          <FontAwesomeIcon icon={faHeart} style={{ color: 'green' }} />
          <FontAwesomeIcon icon={faHeart} style={{ color: 'purple' }} />
          <FontAwesomeIcon icon={faHeart} style={{ color: 'orange' }} />
          <FontAwesomeIcon icon={faHeart} className="text-pink-500" />
          <FontAwesomeIcon icon={faHeart} className="text-indigo-500" />
        </div>
      </section>

      {/* Regular vs Solid Icons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Regular vs Solid Icons</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="text-lg font-medium mb-2">Solid Icons</h3>
            <div className="flex items-center gap-4">
              <FontAwesomeIcon icon={faHeart} />
              <FontAwesomeIcon icon={faStar} />
              <FontAwesomeIcon icon={faUser} />
              <FontAwesomeIcon icon={faEnvelope} />
              <FontAwesomeIcon icon={faBell} />
            </div>
          </div>
          <div>
            <h3 className="text-lg font-medium mb-2">Regular Icons</h3>
            <div className="flex items-center gap-4">
              <FontAwesomeIcon icon={faHeartRegular} />
              <FontAwesomeIcon icon={faStarRegular} />
              <FontAwesomeIcon icon={faUserRegular} />
              <FontAwesomeIcon icon={faEnvelopeRegular} />
              <FontAwesomeIcon icon={faBellRegular} />
            </div>
          </div>
        </div>
      </section>

      {/* Brand Icons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Brand Icons</h2>
        <div className="grid grid-cols-5 gap-4">
          <div className="flex flex-col items-center p-3 border rounded">
            <FontAwesomeIcon icon={faGithub} size="2xl" />
            <span className="mt-2 text-sm">GitHub</span>
          </div>
          <div className="flex flex-col items-center p-3 border rounded">
            <FontAwesomeIcon icon={faTwitter} size="2xl" />
            <span className="mt-2 text-sm">Twitter</span>
          </div>
          <div className="flex flex-col items-center p-3 border rounded">
            <FontAwesomeIcon icon={faFacebook} size="2xl" />
            <span className="mt-2 text-sm">Facebook</span>
          </div>
          <div className="flex flex-col items-center p-3 border rounded">
            <FontAwesomeIcon icon={faInstagram} size="2xl" />
            <span className="mt-2 text-sm">Instagram</span>
          </div>
          <div className="flex flex-col items-center p-3 border rounded">
            <FontAwesomeIcon icon={faLinkedin} size="2xl" />
            <span className="mt-2 text-sm">LinkedIn</span>
          </div>
        </div>
      </section>

      {/* Icons in Buttons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Icons in Buttons</h2>
        <div className="flex flex-wrap gap-4">
          <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded flex items-center gap-2">
            <FontAwesomeIcon icon={faSave} />
            Save
          </button>
          <button className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded flex items-center gap-2">
            <FontAwesomeIcon icon={faPlus} />
            Add New
          </button>
          <button className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded flex items-center gap-2">
            <FontAwesomeIcon icon={faTrash} />
            Delete
          </button>
          <button className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded flex items-center gap-2">
            <FontAwesomeIcon icon={faEdit} />
            Edit
          </button>
        </div>
      </section>

      {/* Animated Icons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Animated Icons</h2>
        <div className="flex items-center gap-4">
          <FontAwesomeIcon icon={faBell} className="animate-bounce" />
          <FontAwesomeIcon icon={faHeart} className="animate-pulse" />
          <FontAwesomeIcon icon={faStar} className="animate-spin" />
          <FontAwesomeIcon icon={faCog} className="animate-spin" />
        </div>
      </section>

      {/* Navigation Icons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Navigation Icons</h2>
        <div className="flex items-center gap-4">
          <button className="p-2 border rounded hover:bg-gray-100">
            <FontAwesomeIcon icon={faArrowLeft} />
          </button>
          <button className="p-2 border rounded hover:bg-gray-100">
            <FontAwesomeIcon icon={faArrowUp} />
          </button>
          <button className="p-2 border rounded hover:bg-gray-100">
            <FontAwesomeIcon icon={faArrowDown} />
          </button>
          <button className="p-2 border rounded hover:bg-gray-100">
            <FontAwesomeIcon icon={faArrowRight} />
          </button>
        </div>
      </section>

      {/* Status Icons */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Status Icons</h2>
        <div className="grid grid-cols-3 gap-4">
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faCheck} className="text-green-500" />
            <span>Success</span>
          </div>
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faTimes} className="text-red-500" />
            <span>Error</span>
          </div>
          <div className="flex items-center gap-2 p-3 border rounded">
            <FontAwesomeIcon icon={faBell} className="text-yellow-500" />
            <span>Warning</span>
          </div>
        </div>
      </section>
    </div>
  );
};

export default FontAwesomeExamples;
